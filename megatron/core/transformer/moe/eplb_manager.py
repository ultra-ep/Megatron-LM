from typing import List, Optional, Tuple, Union, Dict

import torch
from megatron.core import utils
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import ultra_ep
    HAVE_EPLB = True
except ImportError:
    HAVE_EPLB = False


class EPLBManager:
    """Wrapper class for UltraEP manager.

    When pipeline parallelism is enabled (``pp_size > 1``), the manager
    allocates per-microbatch placement / reroute-buffer slots using
    *virtual layer IDs*.  This lets each in-flight micro-batch keep its
    own snapshot of placement state without any extra copies or
    synchronisation; autograd naturally pairs forward ↔ backward via the
    virtual ID saved in ``ctx``.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        ep_group: torch.distributed.ProcessGroup,
    ):
        self.group = ep_group
        self.rank = utils.get_pg_rank(ep_group)
        self.num_ranks = utils.get_pg_size(ep_group)

        self.num_local_master_experts = config.num_moe_experts // self.num_ranks
        self.num_local_redundant_experts = config.moe_num_redundant_experts_per_rank
        self.num_local_physical_experts = self.num_local_master_experts + self.num_local_redundant_experts
        self.num_global_logical_experts = config.num_moe_experts
        self.num_global_physical_experts = self.num_global_logical_experts + self.num_ranks * self.num_local_redundant_experts
        self.local_physical_expert_indices = [
            self.rank * self.num_local_physical_experts + i for i in range(self.num_local_physical_experts)
        ]
        
        self.expert_fc1_numel = 2 * config.hidden_size * config.moe_ffn_hidden_size
        self.expert_fc2_numel = config.hidden_size * config.moe_ffn_hidden_size
        self.expert_total_numel = self.expert_fc1_numel + self.expert_fc2_numel

        # For PP/VPP: peak in-flight micro-batches per layer < pp_size * (vpp_size + 1).
        # For DDP-only (pp_size == 1) this stays 1 → zero overhead.
        pp_size = config.pipeline_model_parallel_size
        vpp_size = config.virtual_pipeline_model_parallel_size
        if vpp_size is None or vpp_size <= 1:
            max_inflight_mbs = pp_size
        else:
            max_inflight_mbs = pp_size * (vpp_size + 1)

        self.max_microbatches = max(1, max_inflight_mbs)

        self.runtime = ultra_ep.Manager(
            group=self.group,
            num_layers=config.num_layers,
            num_local_master_experts=self.num_local_master_experts,
            num_local_redundant_experts=self.num_local_redundant_experts,
            expert_fc1_numel=self.expert_fc1_numel,
            expert_fc2_numel=self.expert_fc2_numel,
            is_train=True,
            explicitly_destroy=False,
            max_microbatches=self.max_microbatches,
        )

        # Mirror placement maps (CPU) from runtime
        self.physical_to_logical_map : torch.Tensor = self.runtime.physical_to_logical_map
        self.logical_to_physical_map : torch.Tensor = self.runtime.logical_to_physical_map
        self.logical_replica_counts : torch.Tensor = self.runtime.logical_replica_counts
        self.physical_to_logical_map_gpu : Optional[torch.Tensor] = None
        self.logical_to_physical_map_gpu : Optional[torch.Tensor] = None
        self.logical_replica_counts_gpu : Optional[torch.Tensor] = None

        # Mirror replica weight and grad buffers (GPU) from runtime
        # Shape: (num_local_redundant_experts, expert_total_numel)
        self.local_replica_weight_buffer : torch.Tensor = self.runtime.local_replica_weight_buffer
        self.local_replica_grad_buffer : torch.Tensor = self.runtime.local_replica_grad_buffer
   
    @torch.no_grad()
    def update_placement(self, layer_id: int, routing_map: torch.Tensor):
        """
        Update the placement of the experts based on the routing stats.
        Uses EPLB-style greedy replication + LPT bin-packing, implemented in C++
        for minimal overhead. Masters remain fixed; only replica slots are updated.
        
        Every rank computes the identical deterministic result, so no broadcast is needed.
        
        Args:
            layer_id: Virtual layer ID (from ``allocate_microbatch_slot``).
            routing_map: ``[num_tokens, num_global_logical_experts]`` bool tensor.
        """
        # Run the C++ placement algorithm (CPU, deterministic)
        self.runtime.update_placement(layer_id, routing_map)
    
    def reroute(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        backend: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand routing map from [num_tokens, num_global_logical_experts] to
        [num_tokens, num_global_physical_experts] using deterministic round-robin
        dispatch across physical replicas.

        For each logical expert l with C_l physical instances, the k-th token
        (ordered by global token index) is assigned to l2p[l, k % C_l].
        This ensures even load distribution across replicas.

        Gradient flow is handled via a custom autograd Function:
          Forward:  expanded_probs[t, phys] = probs[t, logical]  (scatter)
          Backward: grad_probs[t, logical]  = grad_out[t, phys]   (gather)

        Args:
            layer_id: Virtual layer ID (from ``allocate_microbatch_slot``).
            probs: ``[num_tokens, num_global_logical_experts]`` float (GPU).
            routing_map: ``[num_tokens, num_global_logical_experts]`` bool (GPU).
            backend: ``"cuda"`` (fused kernel) or ``"cpu"`` (index arrays).

        Returns:
            ``(expanded_probs, expanded_routing_map)`` in the physical expert space.
        """
        return self.runtime.reroute(
            layer_id, probs, routing_map, backend
        )

    def allocate_microbatch_slot(self, real_layer_id: int) -> int:
        """Allocate a virtual layer ID for the next micro-batch on this layer.

        The returned ID encodes both the real layer and the micro-batch slot.
        Pass this ID (instead of the raw layer number) to ``update_placement``,
        ``reroute``, ``weight_sync``, and ``grad_reduce`` so that each
        in-flight micro-batch uses its own placement / reroute-buffer slot.

        For DDP-only (``max_microbatches == 1``) this returns ``real_layer_id``
        unchanged.
        """
        return self.runtime.allocate_microbatch_slot(real_layer_id)


# Global registry for eplb manager instances
# Shared by all MoE layers within an EP group.
_eplb_manager_registry: Dict[int, EPLBManager] = {}

def get_or_create_eplb_manager(
    config: TransformerConfig,
    ep_group: torch.distributed.ProcessGroup,
) -> EPLBManager:
    """
    Get or create an EPLB manager instance.
    """
    key = id(ep_group)
    global _eplb_manager_registry
    if key not in _eplb_manager_registry:
        _eplb_manager_registry[key] = EPLBManager(config, ep_group)
    return _eplb_manager_registry[key]

def clear_eplb_manager_registry():
    """Clear the global eplb manager registry."""
    global _eplb_manager_registry
    _eplb_manager_registry.clear()