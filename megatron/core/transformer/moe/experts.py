# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import itertools
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.jit import jit_fuser
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules, apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_object_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import internal_api
from megatron.core.transformer.moe.eplb.manager import EPLBManager
from megatron.core.transformer.moe.eplb.shared_grad_buffer import (
    get_or_create_shared_grad_buffer_manager,
    EPLBSharedGradBufferManager,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding

    HAVE_TE = True

except ImportError:

    HAVE_TE = False


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    @internal_api
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        eplb_manager: Optional[EPLBManager] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func
        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            raise ValueError(
                "moe_act recompute for fp8 or fp4 cannot work with the legacy GroupedMLP."
            )

        @jit_fuser
        def activation_func_with_probs(x, probs):
            dtype = x.dtype
            res = self.activation_func(x) * probs
            return res.to(dtype)

        self.activation_func_with_probs = activation_func_with_probs

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        self.dp_group = pg_collection.expt_dp
        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition, self.config.hidden_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
            else:
                # Ensure TP attrs are set even when not initializing
                set_tensor_model_parallel_attributes(
                    tensor=self.weight1, is_parallel=True, dim=1, stride=1
                )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight2, is_parallel=True, dim=0, stride=1
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )
            else:
                # Ensure TP attrs are set even when not initializing
                set_tensor_model_parallel_attributes(
                    tensor=self.weight1, is_parallel=True, dim=1, stride=1
                )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight2, is_parallel=True, dim=0, stride=1
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

        def remove_extra_states_check(self, incompatible_keys):
            """
            Remove _extra_state from unexpected keys.
            These keys are for dist ckpt compatibility with SequentialMLP.
            """
            keys = deepcopy(incompatible_keys.unexpected_keys)
            for key in keys:
                if '_extra_state' in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        assert self.config.bf16, "Currently GroupedGEMM for MoE only supports bf16."
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """
        Maps local expert to global experts.
        The sharded_state_dict for the weight parts are compatible with the SequentialMLP,
        whereas the optimizer states are not due to the limitation from weight transposing.
        That is, for finetuning scenario, the checkpoint is compatible with the SequentialMLP.

        When `singleton_local_shards` metadata flag is True, experts are broken down into
        separate tensors and stored under separate global keys. Additionally, similarly to MLP,
        layers with GLU activations are broken down into separate `w` and `v` tensors.
        """
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()
        dp_rank = self.dp_group.rank()
        num_global_experts = ep_size * self.num_local_experts
        local_expert_indices_offset = ep_rank * self.num_local_experts

        prepend_axis_num = len(sharded_offsets)
        replica_id = (0, 0, dp_rank)

        local_ffn_dim_size = (
            self.weight2.numel() // self.num_local_experts // self.config.hidden_size
        )

        def _break_into_individual_experts(
            experts_ten: torch.Tensor,
            key: str,
            tp_offset: Tuple[int, int, int],
            replica_id: ReplicaId,
        ):
            """Breaks experts into individual tensors and stores them under separate global keys"""
            experts_state = []
            assert len(experts_ten) == self.num_local_experts, (
                experts_ten.shape,
                self.num_local_experts,
            )
            for local_expert_idx, expert_ten in enumerate(experts_ten):
                global_expert_idx = local_expert_indices_offset + local_expert_idx
                expert_key = key.replace(
                    f'{prefix}experts.', f'{prefix}experts.{global_expert_idx}.'
                )
                experts_state.append(
                    ShardedTensor.from_rank_offsets(
                        expert_key,
                        expert_ten.contiguous(),
                        *sharded_offsets,
                        tp_offset,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                )
            return experts_state

        @torch.no_grad()
        def sh_ten_build_fn(
            key: str,
            t: torch.Tensor,
            replica_id: ReplicaId,
            flattened_range: Optional[slice],
            tp_axis: int,
            with_glu: bool,
        ):
            # TODO: write a generic implementation to cover both cases with and without GLU
            if tp_axis == 1:
                # weight1
                if with_glu:
                    last_dim_size = local_ffn_dim_size * 2
                else:
                    last_dim_size = local_ffn_dim_size
                real_shape = (self.num_local_experts, self.config.hidden_size, last_dim_size)
            elif tp_axis == 0:
                # weight2
                real_shape = (self.num_local_experts, local_ffn_dim_size, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if flattened_range is None:
                # weights
                t = t.view(real_shape).transpose(-1, -2)
                # change tp_axis due to the transposing
                tp_axis = 1 - tp_axis
                if with_glu:
                    assert tp_axis == 0, tp_axis
                    if singleton_local_shards:
                        w_tensor, v_tensor = torch.chunk(t, 2, -2)
                        w_key = f'{key}_w'
                        v_key = f'{key}_v'
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': {
                                'w': _break_into_individual_experts(
                                    w_tensor,
                                    w_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                                'v': _break_into_individual_experts(
                                    v_tensor,
                                    v_key,
                                    (prepend_axis_num, tp_rank, tp_size),
                                    replica_id,
                                ),
                            },
                        }
                    else:
                        local_tensors = torch.chunk(t, 2, -2)
                        sub_states = [
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[0].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                            ShardedTensor.from_rank_offsets(
                                key,
                                local_tensors[1].contiguous(),
                                *sharded_offsets,
                                (prepend_axis_num, ep_rank, ep_size),
                                (prepend_axis_num + 1, tp_size + tp_rank, tp_size * 2),
                                replica_id=replica_id,
                                prepend_axis_num=prepend_axis_num,
                            ),
                        ]
                else:
                    if singleton_local_shards:
                        sub_states = {
                            'singleton_local_shards': LocalNonpersistentObject(True),
                            'data': _break_into_individual_experts(
                                t, key, (prepend_axis_num + tp_axis, tp_rank, tp_size), replica_id
                            ),
                        }
                    else:
                        sub_states = ShardedTensor.from_rank_offsets(
                            key,
                            t.contiguous(),
                            *sharded_offsets,
                            (prepend_axis_num, ep_rank, ep_size),
                            (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                            replica_id=replica_id,
                            prepend_axis_num=prepend_axis_num,
                        )
            else:
                if singleton_local_shards:
                    raise NotImplementedError(
                        'flattened_range not supported for'
                        ' GroupedMLP with singleton_local_shards'
                    )
                # flattened optmizer states
                # the non-flattened weight shape is [local_expert_num, hidden_size, ffn_size]
                #
                # For the case without GLU, it is straightforward, we just need to split each
                # expert along the dim-0.
                #
                # For the case with GLU, we need to split the experts along dim-0 and split the
                # two tensors for GLU along dim-2.
                # To split along the non-first dim, we need to chunk the tensor into small pieces,
                # since they belong to different tenors and are interleaved in the flattened space.
                # Refer to the below sketch graph.
                # |................|           |........|........|
                # |............FFFF|           |........|....BBBB|
                # |FFFFFFFFFFFFFFFF|     ->    |AAAAAAAA|BBBBBBBB|
                # |FFFFFFFFFFFFFFFF|           |AAAAAAAA|BBBBBBBB|
                # |FF..............|           |AA......|........|
                # |................|           |........|........|
                #
                # But too many chunks have severe performance issues. We merge these chunks during
                # the save process along with some length information and recover them during the
                # load process.
                assert t.ndim == 1, (key, t.shape)
                if with_glu:
                    non_flat_local_shape = (1, self.config.hidden_size, local_ffn_dim_size)
                    chunk_numel = local_ffn_dim_size
                    sub_states = []
                    start_pos = 0
                    for local_expert_idx in range(self.num_local_experts):
                        first_glu_idx = -1
                        w_start_range = -1
                        v_start_range = -1
                        w_tensors = []
                        v_tensors = []
                        w_lens = []
                        v_lens = []
                        expert_global_idx = local_expert_indices_offset + local_expert_idx
                        for input_dim_idx in range(self.config.hidden_size):
                            for glu_idx in range(2):
                                local_idx = (
                                    local_expert_idx * self.config.hidden_size * 2
                                    + input_dim_idx * 2
                                    + glu_idx
                                )
                                if (
                                    flattened_range.start < chunk_numel * (local_idx + 1)
                                    and flattened_range.stop > chunk_numel * local_idx
                                ):
                                    if first_glu_idx == -1:
                                        first_glu_idx = glu_idx
                                    end_pos = min(
                                        flattened_range.stop,
                                        chunk_numel * (local_idx + 1) - flattened_range.start,
                                    )
                                    local_tensor = t[start_pos:end_pos]
                                    local_flattened_range = slice(
                                        max(0, flattened_range.start - chunk_numel * local_idx),
                                        min(
                                            chunk_numel,
                                            flattened_range.stop - chunk_numel * local_idx,
                                        ),
                                    )
                                    assert (
                                        len(local_tensor)
                                        == local_flattened_range.stop - local_flattened_range.start
                                    )
                                    start_pos += len(local_tensor)
                                    if glu_idx == 0:
                                        w_tensors.append(local_tensor)
                                        w_lens.append(len(local_tensor))
                                        if w_start_range == -1:
                                            w_start_range = max(
                                                0, flattened_range.start - chunk_numel * local_idx
                                            )
                                    else:
                                        v_tensors.append(local_tensor)
                                        v_lens.append(len(local_tensor))
                                        if v_start_range == -1:
                                            v_start_range = max(
                                                0, flattened_range.start - chunk_numel * local_idx
                                            )
                        sub_states.append(
                            {
                                'w_tensors': ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    (
                                        torch.cat(w_tensors, -1)
                                        if len(w_tensors) > 0
                                        else torch.Tensor()
                                    ),
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (
                                        prepend_axis_num,
                                        expert_global_idx,  # pylint: disable=E0606
                                        num_global_experts,
                                    ),
                                    (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size * 2),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=slice(
                                        w_start_range, w_start_range + sum(w_lens)
                                    ),
                                ),
                                'w_lens': LocalNonpersistentObject(w_lens),
                                'v_tensors': ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    (
                                        torch.cat(v_tensors, -1)
                                        if len(v_tensors) > 0
                                        else torch.Tensor()
                                    ),
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (
                                        prepend_axis_num + 1 + tp_axis,
                                        tp_rank + tp_size,
                                        tp_size * 2,
                                    ),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=slice(
                                        v_start_range, v_start_range + sum(v_lens)
                                    ),
                                ),
                                'v_lens': LocalNonpersistentObject(v_lens),
                                'first_glu_idx': LocalNonpersistentObject(first_glu_idx),
                            }
                        )
                else:
                    non_flat_local_shape = (
                        real_shape[0] // self.num_local_experts,
                        *real_shape[1:],
                    )
                    chunk_numel = local_ffn_dim_size * self.config.hidden_size
                    sub_states = []
                    start_pos = 0
                    for local_expert_idx in range(self.num_local_experts):
                        if (
                            flattened_range.start < chunk_numel * (local_expert_idx + 1)
                            and flattened_range.stop > chunk_numel * local_expert_idx
                        ):
                            end_pos = min(
                                flattened_range.stop,
                                chunk_numel * (local_expert_idx + 1) - flattened_range.start,
                            )
                            local_tensor = t[start_pos:end_pos]
                            local_flattened_range = slice(
                                max(0, flattened_range.start - chunk_numel * local_expert_idx),
                                min(
                                    chunk_numel,
                                    flattened_range.stop - chunk_numel * local_expert_idx,
                                ),
                            )
                            assert (
                                len(local_tensor)
                                == local_flattened_range.stop - local_flattened_range.start
                            )
                            start_pos += len(local_tensor)
                            expert_global_idx = local_expert_indices_offset + local_expert_idx
                            sub_states.append(
                                ShardedTensor.from_rank_offsets_flat(
                                    key,
                                    local_tensor,
                                    non_flat_local_shape,
                                    *sharded_offsets,
                                    (prepend_axis_num, expert_global_idx, num_global_experts),
                                    (prepend_axis_num + 1 + tp_axis, tp_rank, tp_size),
                                    replica_id=replica_id,
                                    prepend_axis_num=prepend_axis_num,
                                    flattened_range=local_flattened_range,
                                )
                            )
            return sub_states

        @torch.no_grad()
        def sh_ten_merge_fn(sub_state_dict, tp_axis: int, with_glu: bool):
            if tp_axis == 1:
                # weight1
                weight_shape = (self.config.hidden_size, -1)
            elif tp_axis == 0:
                # weight2
                weight_shape = (-1, self.config.hidden_size)
                assert with_glu == False
            else:
                raise ValueError("tp_axis should be 0 or 1.")
            if isinstance(sub_state_dict, list) and isinstance(sub_state_dict[0], dict):
                # flattened tensor with glu
                res = []
                for local_expert_dict in sub_state_dict:
                    w_tensors = torch.split(
                        local_expert_dict['w_tensors'], local_expert_dict['w_lens']
                    )
                    v_tensors = torch.split(
                        local_expert_dict['v_tensors'], local_expert_dict['v_lens']
                    )
                    first_glu_idx = local_expert_dict['first_glu_idx']
                    if first_glu_idx == 0:
                        res += [
                            x for x in itertools.chain(*itertools.zip_longest(w_tensors, v_tensors))
                        ]
                    else:
                        res += [
                            x for x in itertools.chain(*itertools.zip_longest(v_tensors, w_tensors))
                        ]
                return torch.cat(res)
            elif isinstance(sub_state_dict, list) and sub_state_dict[0].ndim == 1:
                # flattened tensor without glu
                return torch.cat(sub_state_dict)
            else:
                if isinstance(sub_state_dict, dict):
                    assert sub_state_dict['singleton_local_shards']
                    if with_glu:
                        assert isinstance(sub_state_dict['data'], dict)
                        sub_state_dict = torch.cat(
                            (
                                torch.stack(sub_state_dict['data']['w']),
                                torch.stack(sub_state_dict['data']['v']),
                            ),
                            dim=-2,
                        )
                    else:
                        assert isinstance(sub_state_dict['data'], list)
                        sub_state_dict = torch.stack(sub_state_dict['data'])
                else:
                    if with_glu:
                        sub_state_dict = torch.cat(sub_state_dict, -2)
                return sub_state_dict.transpose(-1, -2).reshape(weight_shape)

        state_dict = self.state_dict(prefix='', keep_vars=True)
        for name, tensor in state_dict.items():
            if name == 'weight1':
                tp_axis = 1
                with_glu = self.config.gated_linear_unit
                wkey = f'{prefix}experts.linear_fc1.weight'
            else:
                tp_axis = 0
                with_glu = False
                wkey = f'{prefix}experts.linear_fc2.weight'

            this_replica_id = list(copy.deepcopy(replica_id))
            flattened_range = None

            sharded_state_dict[f'{prefix}{name}'] = ShardedTensorFactory(
                wkey,
                tensor,
                partial(sh_ten_build_fn, tp_axis=tp_axis, with_glu=with_glu),
                partial(sh_ten_merge_fn, tp_axis=tp_axis, with_glu=with_glu),
                tuple(this_replica_id),
                flattened_range=flattened_range,
            )

        replica_id = (0, tp_rank, dp_rank)
        # Add fake _extra_state to be compatible with SequentialMLP
        for expert_local_idx in range(self.num_local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            if singleton_local_shards:
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )
            for mod in ['linear_fc1', 'linear_fc2']:
                if singleton_local_shards:
                    expert_key = f'{prefix}experts.{expert_global_idx}.{mod}._extra_state'
                else:
                    expert_key = f'{prefix}experts.{mod}._extra_state'
                sharded_state_dict[f'{prefix}expert{expert_global_idx}.{mod}._extra_state'] = (
                    make_sharded_object_for_checkpoint(
                        None, expert_key, expert_sharded_offsets, replica_id
                    )
                )

        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    
    When EPLB is enabled, this class handles both original experts and replica
    experts. The number of local experts includes replicas in that case.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    @internal_api
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
        eplb_manager: Optional[EPLBManager] = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        
        # EPLB support
        self.eplb_manager = eplb_manager
        self.eplb_enabled = eplb_manager is not None
        if self.eplb_enabled:
            self.num_local_physical_experts = eplb_manager.num_local_physical_experts
            assert self.num_local_physical_experts == self.num_local_experts, \
                "num_local_physical_experts should be equal to num_local_experts"
            self.num_local_master_experts = eplb_manager.num_local_master_experts
            self.num_local_replica_experts = eplb_manager.num_redundant_per_rank
            assert self.num_local_master_experts + self.num_local_replica_experts == self.num_local_physical_experts, \
                "num_local_master_experts + num_local_replica_experts should be equal to num_local_physical_experts"
        
        self.input_size = self.config.hidden_size
        assert not (
            self.config.add_bias_linear and config.bias_dropout_fusion
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.moe_ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.num_local_experts,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name='fc1',
            pg_collection=pg_collection,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = build_module(submodules.activation_func, config=self.config)
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.num_local_experts,
            self.config.moe_ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
            pg_collection=pg_collection,
        )

        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in self.config.offload_modules
        )

        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in self.config.offload_modules
        )

        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc2)

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1 and not (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc1)

        if self.config.fp8 or self.config.fp4:
            assert HAVE_TE, "FP8 and FP4 requires TE."
            self.quantization_padding = Fp8Padding(self.num_local_experts)
            self.quantization_unpadding = Fp8Unpadding(self.num_local_experts)

        # Mark replica expert parameters to exclude them from optimizer state allocation.
        # Replicas only need gradient buffers for accumulation; their optimizer states would
        # be wasted since weights are always synchronized from masters after optimizer step.
        if self.eplb_enabled:
            self._mark_replica_parameters()
            # Initialize EPLB gradient aggregation state
            self._eplb_replica_grad_reduce_handle = None
            
            # Get or create shared gradient buffer manager for EPLB replicas
            # This is shared across all MoE layers to save memory
            self._shared_grad_buffer_manager: Optional[EPLBSharedGradBufferManager] = None
            self._shared_grad_buffer_initialized = False

    def _mark_replica_parameters(self):
        """Mark replica expert parameters with is_eplb_replica=True attribute.
        
        This allows the optimizer to skip allocating states for these parameters,
        saving GPU memory. Replica parameters still compute gradients (which are
        aggregated to masters), but don't need their own optimizer states.
        """
        num_local_master = self.eplb_manager.num_local_master_experts
        num_redundant = self.eplb_manager.num_redundant_per_rank
        
        for linear_module in [self.linear_fc1, self.linear_fc2]:
            for replica_idx in range(num_redundant):
                local_physical_idx = num_local_master + replica_idx
                
                # Mark weight parameter
                replica_weight = getattr(linear_module, f'weight{local_physical_idx}', None)
                if replica_weight is not None:
                    # Mark as replica for skipping optimizer state and main_grad buffer
                    setattr(replica_weight, 'is_eplb_replica', True)
    
    def initialize_shared_grad_buffer(self, layer_name: str = ""):
        """Initialize shared gradient buffer for EPLB replica expert parameters.
        
        This method should be called after model construction is complete and before
        training starts. It sets up a shared gradient buffer that is reused across
        all MoE layers to save memory.
        
        The shared buffer manager is obtained/created once and shared across all
        layers with the same configuration. Each layer's replica params get views
        into this shared buffer as their main_grad.
        
        Args:
            layer_name: Name of this layer for logging purposes
        """
        if not self.eplb_enabled or self._shared_grad_buffer_initialized:
            return
        
        num_redundant = self.eplb_manager.num_redundant_per_rank
        if num_redundant == 0:
            self._shared_grad_buffer_initialized = True
            return
        
        # Get weight shapes from first replica expert
        num_local_master = self.eplb_manager.num_local_master_experts
        fc1_weight = getattr(self.linear_fc1, f'weight{num_local_master}')
        fc2_weight = getattr(self.linear_fc2, f'weight{num_local_master}')
        
        # Determine gradient dtype (float32 for gradient accumulation fusion)
        grad_dtype = torch.float32 if self.config.gradient_accumulation_fusion else fc1_weight.dtype
        
        # Get or create shared buffer manager
        self._shared_grad_buffer_manager = get_or_create_shared_grad_buffer_manager(
            ep_group=self.ep_group,
            num_redundant_per_rank=num_redundant,
            fc1_weight_shape=fc1_weight.shape,
            fc2_weight_shape=fc2_weight.shape,
            grad_dtype=grad_dtype,
        )
        
        # Assign main_grad to replica params from shared buffer
        self._shared_grad_buffer_manager.assign_main_grad_to_replica_params(
            linear_fc1=self.linear_fc1,
            linear_fc2=self.linear_fc2,
            num_local_master=num_local_master,
            layer_name=layer_name,
        )
        
        self._shared_grad_buffer_initialized = True
                
    def _init_eplb_index_mappings(self):
        """Pre-compute index mappings as GPU tensors to avoid ALL D2H transfers.
        
        This stores slices of the physical_to_logical_map as GPU tensors, which can
        be used for advanced indexing (index_copy_, index_select) during backward.
        """
        if not self.eplb_manager._placement_initialized:
            return
        
        ep_rank = self.eplb_manager.ep_rank
        num_local_master = self.eplb_manager.num_local_master_experts
        num_redundant = self.eplb_manager.num_redundant_per_rank
        num_local_physical = self.eplb_manager.num_local_physical_experts
        
        # Keep as GPU tensor
        physical_to_logical = self.eplb_manager.placement.physical_to_logical_map
        
        # Pre-compute replica logical indices: GPU tensor [num_redundant]
        replica_start = ep_rank * num_local_physical + num_local_master
        replica_end = replica_start + num_redundant
        self._eplb_replica_logical_indices = physical_to_logical[replica_start:replica_end].long()
        
        # Pre-compute master logical indices: GPU tensor [num_local_master]
        master_start = ep_rank * num_local_physical
        master_end = master_start + num_local_master
        self._eplb_master_logical_indices = physical_to_logical[master_start:master_end].long()
        
    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.tolist()
        if self.config.fp8 or self.config.fp4:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.offload_expert_fc1:
            permuted_local_hidden_states = fine_grained_offloading_group_start(
                permuted_local_hidden_states, name="expert_fc1"
            )
        with get_fine_grained_offloading_context(self.offload_expert_fc1):
            fc1_output, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output, bias_parallel = fine_grained_offloading_group_commit(
                fc1_output,
                bias_parallel,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert bias_parallel is None
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                if self.config.gated_linear_unit:

                    def glu(x):
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        if (val := self.config.activation_func_clamp_value) is not None:
                            x_glu = x_glu.clamp(min=None, max=val)
                            x_linear = x_linear.clamp(min=-val, max=val)
                        return self.config.activation_func(x_glu) * (
                            x_linear + self.config.glu_linear_offset
                        )

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
            return intermediate_parallel

        if self.offload_moe_act:
            fc1_output = fine_grained_offloading_group_start(fc1_output, name="moe_act")

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_moe_act):
                bias_act_output = self.activation_checkpoint.checkpoint(
                    bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with get_fine_grained_offloading_context(self.offload_moe_act):
                bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)

        output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)
        if self.offload_moe_act:
            (output,) = fine_grained_offloading_group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )

        # upad and concat the output
        if self.config.fp8 or self.config.fp4:
            output = self.quantization_unpadding(output, actual_tokens_per_expert)

        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)
        output_bias = None

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(
                module, f'{name}.', sharded_offsets, metadata, tp_group=self.tp_group
            )
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = self.ep_group.size() * self.num_local_experts
                local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, '', f'{prefix}experts.')
            else:
                # Add prefix here to match sequential's keys
                replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in TEGroupedMLP.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on the linear layers in reverse order (fc2 followed by fc1).
        If an error occurs during execution, it is caught and re-raised with a
        descriptive message.
        """
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()

    def _get_param_grad(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get gradient from parameter.
        
        With gradient_accumulation_fusion (default), gradients are written directly to main_grad.
        Without fusion, gradients are in param.grad and copied to main_grad by DDP hooks.
        """
        assert hasattr(param, 'main_grad') and param.main_grad is not None, "main_grad is not available"
        return param.main_grad
    
    def _zero_param_grad(self, param: torch.nn.Parameter):
        """Zero out parameter's gradient buffer.
        Note: Prefer using self._shared_grad_buffer_manager.zero_grad() for bulk zero-out,
        instead of this method for per-parameter zero-out.
        """
        assert hasattr(param, 'main_grad') and param.main_grad is not None, "main_grad is not available"
        param.main_grad.zero_()

    def start_replica_grad_reduce(self, async_op: bool = True):
        """Start async all-reduce of replica gradients to master experts.
        
        This should be called after all replica expert gradients are computed in the 
        current microbatch. It initiates async communication that can overlap with 
        subsequent backward computation.
        
        Performance optimizations:
        - ZERO D2H/H2D: Uses GPU tensors for all index mapping operations
        - Bulk operations: Leverages shared buffer's contiguous layout for stacked grads
        - Bulked and async zero operation: Zeros entire shared gradient buffer at once
        - Fused buffer: Single all-reduce for both fc1 and fc2 gradients
        - Tensorized movement: Uses index_copy_ to fill global buffers on GPU
        
        Args:
            async_op: If True, return immediately with async handle. If False, block.
            
        Returns:
            Communication handle if async_op=True, else None
        """
        if not self.eplb_enabled or self.eplb_manager is None:
            return None
        
        if not self.eplb_manager._placement_initialized:
            return None
        
        # Initialize index mappings on first call
        self._init_eplb_index_mappings()
        
        ep_group = self.eplb_manager.ep_group
        num_redundant = self.eplb_manager.num_redundant_per_rank
        
        if num_redundant == 0:
            return None
        
        # Ensure shared buffer manager is available
        if self._shared_grad_buffer_manager is None:
            raise RuntimeError(
                "Shared grad buffer manager not initialized. "
                "Call initialize_shared_grad_buffer() first."
            )
        
        # Initialize all-reduce buffer on first call
        num_global_logical = self.eplb_manager.num_global_logical_experts
        grad_reduce_buffer = self._shared_grad_buffer_manager.get_or_create_global_logical_experts_grad_buffer(num_global_logical)

        # Zero out the shared buffer for replica gradients
        grad_reduce_buffer.zero_()

        # Get fc1 and fc2 replica gradients from shared buffer (already stacked and contiguous)
        fc1_replica_grads = self._shared_grad_buffer_manager.get_fc1_stacked_grads()
        fc2_replica_grads = self._shared_grad_buffer_manager.get_fc2_stacked_grads()
        
        # Flatten and concatenate fc1 and fc2 gradients: [num_redundant, fc1_numel + fc2_numel]
        fc1_flat = fc1_replica_grads.view(num_redundant, -1)
        fc2_flat = fc2_replica_grads.view(num_redundant, -1)
        fused_replica_grads = torch.cat([fc1_flat, fc2_flat], dim=1)
        
        # Bulk index_copy_ into fused buffer
        grad_reduce_buffer.index_copy_(
            0, self._eplb_replica_logical_indices, fused_replica_grads
        )
        
        # Start single async all-reduce for the fused buffer
        handle = torch.distributed.all_reduce(
            grad_reduce_buffer, group=ep_group, async_op=async_op
        )
        
        # Allow overlapping of zero-out with all-reduce
        self._shared_grad_buffer_manager.zero_grad()
                
        if async_op:
            self._eplb_replica_grad_reduce_handle = handle
            return handle
        return None
    
    def finish_replica_grad_reduce(self):
        """Finish async replica gradient reduction and add to master gradients.
        
        This waits for the async all-reduce to complete and adds the aggregated
        replica gradient contributions to the corresponding master expert gradients.
        
        Uses ZERO D2H/H2D optimizations:
        - index_select gathers all master contributions at once on GPU
        - Extracts fc1 and fc2 portions from fused buffer
        """
        if not self.eplb_enabled or self.eplb_manager is None:
            return
        
        if self._eplb_replica_grad_reduce_handle is not None:
            # Wait for the async all-reduce to complete
            self._eplb_replica_grad_reduce_handle.wait()
            self._eplb_replica_grad_reduce_handle = None
        
        num_local_master = self.eplb_manager.num_local_master_experts

        fc1_shape = self._shared_grad_buffer_manager.fc1_weight_shape
        fc2_shape = self._shared_grad_buffer_manager.fc2_weight_shape
        fc1_numel = self._shared_grad_buffer_manager.fc1_numel_per_expert
        
        grad_reduce_buffer = self._shared_grad_buffer_manager.get_or_create_global_logical_experts_grad_buffer()
        
        # Gather all local master contributions at once: [num_local_master, fc1_numel + fc2_numel]
        master_contribs = grad_reduce_buffer.index_select(0, self._eplb_master_logical_indices)
        
        # Split into fc1 and fc2 portions
        fc1_contribs = master_contribs[:, :fc1_numel].view(num_local_master, *fc1_shape)
        fc2_contribs = master_contribs[:, fc1_numel:].view(num_local_master, *fc2_shape)
        
        # Add fc1 contributions to master gradients
        for i in range(num_local_master):
            master_weight = getattr(self.linear_fc1, f'weight{i}')
            master_grad = self._get_param_grad(master_weight)
            if master_grad is not None:
                master_grad.add_(fc1_contribs[i].to(master_grad.dtype))
        
        # Add fc2 contributions to master gradients
        for i in range(num_local_master):
            master_weight = getattr(self.linear_fc2, f'weight{i}')
            master_grad = self._get_param_grad(master_weight)
            if master_grad is not None:
                master_grad.add_(fc2_contribs[i].to(master_grad.dtype))
            
    def synchronize_replica_weights_with_master(self):
        """Synchronize replica weights from their source originals.
        
        This will refresh all replica weights from source master experts. Should be
        used after optimizer step or start of training.
        
        Performance optimizations:
        - ZERO D2H/H2D: All index math and copies happen on GPU
        - Bulk movement: Uses advanced indexing to copy all replica weights at once
        """
        if not self.eplb_enabled or self.eplb_manager is None:
            return
        
        if not self.eplb_manager._placement_initialized:
            return
        
        ep_group = self.eplb_manager.ep_group
        ep_size = self.eplb_manager.ep_size
        ep_rank = self.eplb_manager.ep_rank
        num_local_master = self.eplb_manager.num_local_master_experts
        num_redundant = self.eplb_manager.num_redundant_per_rank
        num_local_physical = self.eplb_manager.num_local_physical_experts
        
        if num_redundant == 0:
            return
        
        physical_to_logical = self.eplb_manager.placement.physical_to_logical_map
        logical_to_physical = self.eplb_manager.placement.logical_to_physical_map
        
        # Pre-compute source coordinates for all local replicas on GPU
        replica_start = ep_rank * num_local_physical + num_local_master
        replica_end = replica_start + num_redundant
        replica_logical_indices = physical_to_logical[replica_start:replica_end].long()
        
        # First entry in logical_to_physical is always the master expert
        master_physical_indices = logical_to_physical[replica_logical_indices, 0].long()
        
        source_ranks = master_physical_indices // num_local_physical
        source_local_indices = master_physical_indices % num_local_physical
        
        # Process each linear layer (fc1 and fc2)
        for linear_module in [self.linear_fc1, self.linear_fc2]:
            # 1. Stack local master weights and gather
            master_weights = torch.stack([
                getattr(linear_module, f'weight{i}').data 
                for i in range(num_local_master)
            ], dim=0)
            
            gathered_list = [torch.empty_like(master_weights) for _ in range(ep_size)]
            torch.distributed.all_gather(gathered_list, master_weights, group=ep_group)
            all_master_weights = torch.stack(gathered_list, dim=0)
            
            # 2. Bulk copy all replica weights at once using advanced indexing
            # Collect replica weight references
            replica_weights_ref = [
                getattr(linear_module, f'weight{num_local_master + i}')
                for i in range(num_redundant)
            ]
            
            # Create a single view of all local replica weights for the bulk copy
            # We use stack to create a temporary buffer for the copy operation
            # then copy back to individual parameters. 
            # *In a future refactor, TEGroupedLinear could use a single weight tensor.
            all_local_replicas = torch.stack([w.data for w in replica_weights_ref])
            
            # GPU-to-GPU advanced indexing copy
            all_local_replicas.copy_(all_master_weights[source_ranks, source_local_indices])
            
            # Copy back from stacked buffer to individual parameters
            for i, w in enumerate(replica_weights_ref):
                w.data.copy_(all_local_replicas[i])


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    @internal_api
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)

    def _pad_tensor_for_quantization(self, hidden, probs):
        """Padding tensor shape to multiples of 16/32."""
        actual_num_tokens = hidden.shape[0]
        divisor = get_align_size_for_quantization(self.config)
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
            pad_probs = torch.zeros(padded_num_tokens, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_probs), dim=0)
        return hidden, probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the SequentialMLP."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if self.config.fp8 or self.config.fp4:
                hidden, probs = self._pad_tensor_for_quantization(
                    permuted_local_hidden_states, permuted_probs
                )
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](
                    permuted_local_hidden_states, permuted_probs
                )

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
            probs_list = torch.split(permuted_probs, tokens_per_expert)

            output_local_list = []

            for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
                if self.config.fp8 or self.config.fp4:
                    hidden, probs = self._pad_tensor_for_quantization(tokens, probs)
                    output, output_bias = expert(hidden, probs)
                    output = output[: tokens.shape[0]]
                else:
                    output, output_bias = expert(tokens, probs)
                output_local_list.append(output)

            output_local = torch.cat(output_local_list, dim=0)
            output_bias_local = None
            # Note: if bias is enabled on experts, it is already added to the output at this point
            return output_local, output_bias_local

    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        for expert in self.local_experts:
            expert.backward_dw()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            if singleton_local_shards:
                expert_sharded_prefix = f'{prefix}experts.{expert_global_idx}.'
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f'{prefix}experts.'
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'

                sh_ten.replica_id = (*replica_id[:2], self.dp_group.rank())

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
