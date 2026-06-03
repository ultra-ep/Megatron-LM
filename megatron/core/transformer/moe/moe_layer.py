# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.expert_load_recorder import (
    get_or_create_expert_load_recorder,
)
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.eplb_manager import HAVE_EPLB, get_or_create_eplb_manager

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class _EPLBReplicaGradReduceStartFunction(torch.autograd.Function):
    """Autograd function to trigger EPLB replica gradient reduction during backward.
    
    This function wraps the MoE INPUT (not output) so that its backward fires AFTER
    the MoE layer's backward is complete. The backward execution order is:
    
    Forward:  input → [wrap] → MoE forward → output
    Backward: output_grad → MoE backward → [wrap backward] → input_grad
    
    By wrapping the input, the wrapper's backward fires after MoE backward completes,
    which means replica gradients are already in main_grad and ready for reduction.
    
    The reduction happens for EVERY microbatch (unlike DDP which only reduces on
    the last microbatch) because replica gradients must be aggregated to master
    experts immediately for correct gradient accumulation.
    """
    
    @staticmethod
    def forward(ctx, hidden_states, moe_layer, virtual_layer_id):
        ctx.moe_layer = moe_layer
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states
    
    @staticmethod
    def backward(ctx, grad_output):
        ctx.moe_layer._eplb_start_grad_reduce(
            virtual_layer_id=ctx.virtual_layer_id
        )
        return grad_output, None, None


class _EPLBReplicaGradReduceFinishFunction(torch.autograd.Function):
    """Autograd function to finish EPLB replica gradient reduction during backward.
    
    This function wraps the MoE INPUT (not output) so that its backward fires AFTER
    the MoE layer's backward is complete. The backward execution order is:
    
    Forward:  input → [wrap] → MoE forward → output
    Backward: output_grad → MoE backward → [wrap backward] → input_grad
    
    By wrapping the input, the wrapper's backward fires after MoE backward completes,
    which means replica gradients are already in main_grad and ready for reduction.
    
    The reduction happens for EVERY microbatch (unlike DDP which only reduces on
    the last microbatch) because replica gradients must be aggregated to master
    experts immediately for correct gradient accumulation.
    """
    
    @staticmethod
    def forward(ctx, hidden_states, moe_layer):
        """Forward pass: save moe_layer reference and pass through input."""
        ctx.moe_layer = moe_layer
        return hidden_states
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: finish EPLB replica gradient reduction.
        """        
        ctx.moe_layer._eplb_finish_grad_reduce()
        return grad_output, None


class _EPLBWeightSyncFunction(torch.autograd.Function):
    """Autograd function to sync EPLB replica weights.
    Used in bprop w/o recompute to sync replica weights with masters.
    """

    @staticmethod
    def forward(ctx, hidden_states, moe_layer, virtual_layer_id):
        ctx.moe_layer = moe_layer
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.moe_layer.eplb_manager is not None:
            ctx.moe_layer.eplb_manager.runtime.weight_sync(
                layer_id=ctx.virtual_layer_id,
                async_finish=False,
            )
        return grad_output, None, None


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.is_full_recompute = (
            config.recompute_granularity == 'full'
            and config.recompute_method == 'uniform'
            and config.recompute_num_layers == 1
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )
        self.num_local_master_experts = self.num_local_experts
        self.local_master_expert_indices = self.local_expert_indices

        # Initialize EPLB Manager (if enabled)
        self.eplb_enabled = config.moe_enable_eplb
        self.eplb_manager = None
        
        if self.eplb_enabled:
            if not HAVE_EPLB:
                raise ImportError(
                    "EPLB is enabled but eplb module could not be imported. "
                    "Please ensure megatron.core.transformer.moe.eplb is available."
                )
            self.eplb_manager = get_or_create_eplb_manager(
                config=config,
                ep_group=self.ep_group,
            )
            # Number of experts to compute = local masters + local replicas
            self.num_local_physical_experts = self.eplb_manager.num_local_physical_experts
            self.num_global_physical_experts = self.eplb_manager.num_global_physical_experts
            self.local_physical_expert_indices = self.eplb_manager.local_physical_expert_indices
            # Event handles
            self._eplb_grad_reduce_event_handle = None
            self._eplb_weight_sync_event_handle = None

        else:
            self.num_local_physical_experts = self.num_local_master_experts
            self.num_global_physical_experts = self.config.num_moe_experts
            self.local_physical_expert_indices = self.local_master_expert_indices

        self.expert_load_recorder = get_or_create_expert_load_recorder(
            ep_group=self.ep_group,
            num_global_physical_experts=self.num_global_physical_experts,
            num_local_physical_experts=self.num_local_physical_experts,
        )
        if self.expert_load_recorder is not None:
            self.expert_load_recorder.register_layer(self.layer_number)

        # Initialize router
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)
        self.tp_group = pg_collection.tp
        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            # All Gather dispatcher does not support EPLB
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_physical_experts,
                self.local_physical_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
                num_global_physical_experts=self.num_global_physical_experts,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_physical_experts,
                self.local_physical_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
                num_global_physical_experts=self.num_global_physical_experts,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        # When EPLB is enabled, we need to allocate extra capacity for replica experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_physical_experts,  # Includes replicas when EPLB is enabled
            self.config,
            pg_collection=pg_collection,
        )
        self._eplb_master_ptrs_registered = not self.eplb_enabled
        if self.eplb_enabled:
            assert isinstance(self.experts, TEGroupedMLP), \
                f"experts must be a TEGroupedMLP when EPLB is enabled, but got {type(self.experts)}"
            # Phase 1: mark replicas and set their data/grad to shared UltraEP buffers.
            self._eplb_register_redundant_experts()
            # Phase 2 (_eplb_register_master_experts) must be called after DDP init.

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts,
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

    def _eplb_register_redundant_experts(self):
        """Phase 1: Register replica expert weights and grads with shared UltraEP buffers.

        Called during model initialization, BEFORE DDP / _ParamAndGradBuffer init.

        This method:
            - Marks replica weight parameters with ``is_eplb_replica = True`` so that
              _ParamAndGradBuffer skips allocating weight and gradient buffer space
              for them (they use cross-layer shared buffers from UltraEP instead).
            - Re-points each replica weight's ``.data`` and ``.main_grad`` to the
              corresponding views in UltraEP's shared buffers.  Because
              _ParamAndGradBuffer skips ``is_eplb_replica`` params, these assignments
              are never overwritten.

        Master expert pointer registration (``construct_local_master_ptr_pool``)
        is deferred to :meth:`_eplb_register_master_experts`, which MUST be called
        after DDP initialization completes (when ``main_grad`` has been assigned to
        master weights by ``_ParamAndGradBuffer``).
        """
        num_local_master = self.eplb_manager.num_local_master_experts
        num_local_redundant = self.eplb_manager.num_local_redundant_experts
        num_local_physical = num_local_master + num_local_redundant
        expert_fc1_numel = self.eplb_manager.expert_fc1_numel
        expert_fc2_numel = self.eplb_manager.expert_fc2_numel
        expert_total_numel = self.eplb_manager.expert_total_numel
        local_replica_weight_buffer = self.eplb_manager.local_replica_weight_buffer
        local_replica_grad_buffer = self.eplb_manager.local_replica_grad_buffer

        for module_idx, linear_module in enumerate([self.experts.linear_fc1, self.experts.linear_fc2]):
            expert_weight0 = getattr(linear_module, 'weight0', None)
            if module_idx == 0:
                assert expert_fc1_numel == expert_weight0.numel()
                module_shape = expert_weight0.shape
                expert_data_range = slice(0, expert_fc1_numel)
            else:
                assert expert_fc2_numel == expert_weight0.numel()
                module_shape = expert_weight0.shape
                expert_data_range = slice(expert_fc1_numel, expert_total_numel)

            # Only process replica experts (indices >= num_local_master)
            for expert_idx in range(num_local_master, num_local_physical):
                local_replica_offset = expert_idx - num_local_master
                replica_weight = getattr(linear_module, f'weight{expert_idx}', None)
                assert replica_weight is not None, (
                    f"weight{expert_idx} is not found in {linear_module}"
                )

                # Mark as EPLB replica so that:
                #   - _ParamAndGradBuffer skips weight & grad buffer allocation
                #   - Distributed optimizer skips optimizer state allocation
                setattr(replica_weight, 'is_eplb_replica', True)

                # Re-point .data and .main_grad to views in UltraEP's cross-layer
                # shared buffers.  _ParamAndGradBuffer will skip is_eplb_replica
                # params, so these assignments are preserved.
                replica_weight.data = (
                    local_replica_weight_buffer[local_replica_offset, expert_data_range]
                    .view(module_shape)
                )
                replica_weight.main_grad = (
                    local_replica_grad_buffer[local_replica_offset, expert_data_range]
                    .view(module_shape)
                )

        # Inform TEGroupedMLP how many master experts to include in checkpoints.
        # This is used by TEGroupedMLP.sharded_state_dict to filter out replicas
        # and fix the global shape / offset metadata for master expert tensors.
        self.experts.num_local_master_experts = num_local_master

        self._eplb_master_ptrs_registered = False

    def _eplb_register_master_experts(self):
        """Phase 2: Register master expert weight/grad pointers with UltraEP runtime.

        MUST be called AFTER DDP initialization (``_ParamAndGradBuffer`` has mapped
        master expert weights to contiguous ``param_data`` and assigned ``main_grad``
        from ``grad_data``).

        This method collects the **final** ``.data`` and ``.main_grad`` tensor
        pointers for master experts and passes them to UltraEP's
        ``construct_local_master_ptr_pool`` so that ``weight_sync`` and
        ``grad_reduce`` operations can locate the correct device memory.
        """
        if self._eplb_master_ptrs_registered:
            return

        num_local_master = self.eplb_manager.num_local_master_experts

        master_fc1_weights = []
        master_fc2_weights = []
        master_fc1_grads = []
        master_fc2_grads = []

        for module_idx, linear_module in enumerate([self.experts.linear_fc1, self.experts.linear_fc2]):
            for expert_idx in range(num_local_master):
                master_weight = getattr(linear_module, f'weight{expert_idx}', None)
                assert master_weight is not None, (
                    f"weight{expert_idx} is not found in {linear_module}"
                )
                assert hasattr(master_weight, 'main_grad'), (
                    f"weight{expert_idx}.main_grad not found in {linear_module}. "
                    f"_eplb_register_master_experts() must be called after DDP initialization."
                )

                if module_idx == 0:
                    master_fc1_weights.append(master_weight.data)
                    master_fc1_grads.append(master_weight.main_grad)
                else:
                    master_fc2_weights.append(master_weight.data)
                    master_fc2_grads.append(master_weight.main_grad)

        self.eplb_manager.runtime.construct_local_master_ptr_pool(
            layer_id=self.layer_number,
            fc1_weights=master_fc1_weights,
            fc2_weights=master_fc2_weights,
            fc1_grads=master_fc1_grads,
            fc2_grads=master_fc2_grads,
        )
        self._eplb_master_ptrs_registered = True
    
    def _eplb_start_grad_reduce(
        self,
        virtual_layer_id: int,
        async_finish: bool = True,
    ):
        self._eplb_grad_reduce_event_handle = (
            self.eplb_manager.runtime.grad_reduce(
                layer_id=virtual_layer_id,
                async_finish=async_finish,
            )
        )

    def _eplb_finish_grad_reduce(self):
        if self._eplb_grad_reduce_event_handle is not None:
            self._eplb_grad_reduce_event_handle.current_stream_wait()
            self._eplb_grad_reduce_event_handle = None

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(self, hidden_states: torch.Tensor):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping.
        """
        probs, routing_map = self.router(hidden_states)
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher. The original hidden states are returned as a residual connection.
        """
        residual = hidden_states
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def _should_capture_expert_load(self) -> bool:
        """Capture only on real forwards, not CUDA-graph replay or recompute."""
        if self.expert_load_recorder is None:
            return False
        if is_graph_capturing() or not self.cudagraph_tensor_store.is_empty():
            return False
        if self.training and (
            self.moe_layer_recompute or self.config.recompute_granularity == "full"
        ):
            return not torch.is_grad_enabled()
        return True

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    def routed_experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # Lazily finalize EPLB master pointer registration after DDP init.
        # This is a safety net; callers should ideally invoke
        # _eplb_register_master_experts() explicitly after DDP construction.
        if self.eplb_enabled and not self._eplb_master_ptrs_registered:
            self._eplb_register_master_experts()

        # Allocate a virtual layer ID for this micro-batch OUTSIDE custom_forward
        # so that tensor_parallel.checkpoint (activation recompute) captures the
        # same ID via closure — both the original forward and the recompute
        # forward use identical placement / reroute-buffer slots.
        virtual_layer_id = None
        if self.eplb_enabled:
            virtual_layer_id = self.eplb_manager.allocate_microbatch_slot(
                self.layer_number
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            try:
                shared_expert_output = self.shared_experts_compute(hidden_states)
                probs, routing_map = self.route(hidden_states)

                # EPLB: expand routing map to include replica assignments
                if self.eplb_enabled and self.eplb_manager is not None:
                    # Update replica placement based on real-time expert loads.
                    self.eplb_manager.update_placement(virtual_layer_id, routing_map)
                    # Sync replica weights with masters.
                    self._eplb_weight_sync_event_handle = (
                        self.eplb_manager.runtime.weight_sync(
                            layer_id=virtual_layer_id, async_finish=True
                        )
                    )
                    # Reroute tokens to replica experts.
                    probs, routing_map = self.eplb_manager.reroute(
                        virtual_layer_id, probs, routing_map
                    )

                hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
            except MoECudaGraphPartialCaptureSignal as e:
                # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                # It means we should early-return from the MoE layer forward pass.
                # This happens when we are partially capturing the CUDA graph of the MoE layer,
                # like cuda_graph_scope=["moe_router", "moe_preprocess"].
                # We need to return the intermediate tensors as CUDA graph outputs.
                return e.get_early_return_outputs(hidden_states, shared_expert_output)

            if self._should_capture_expert_load():
                self.expert_load_recorder.capture(self.layer_number, routing_map)

            # EPLB: Wrap input with autograd function to trigger replica gradient reduction
            # during backward pass. By wrapping the INPUT, the wrapper's backward fires AFTER
            # the MoE layer's backward is complete (when gradients are in main_grad).
            # This ensures replica gradients are reduced after each microbatch's backward
            # (not just the last one like DDP overlap_grad_reduce).
            if self.eplb_enabled:
                hidden_states = _EPLBReplicaGradReduceStartFunction.apply(
                    hidden_states, self, virtual_layer_id
                )
                if self._eplb_weight_sync_event_handle is not None:
                    self._eplb_weight_sync_event_handle.current_stream_wait()
                    self._eplb_weight_sync_event_handle = None

            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
            output = self.combine(output, shared_expert_output)
            if self.eplb_enabled and not (self.moe_layer_recompute or self.is_full_recompute):
                # Re-sync replica weights with masters in bprop w/o recompute.
                output = _EPLBWeightSyncFunction.apply(
                    output, self, virtual_layer_id
                )
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                outputs = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            outputs = custom_forward(hidden_states)

        return outputs

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
