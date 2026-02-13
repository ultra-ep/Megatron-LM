from typing import List, Optional, Tuple, Union, Dict

import torch
import random
from megatron.core import utils
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import ultra_ep
    from ultra_ep.util import setup_placement_random
    HAVE_EPLB = True
except ImportError:
    HAVE_EPLB = False


class EPLBManager:
    """
    Wrapper class for UltraEP manager.
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
        
        self.placement_strategy = config.moe_eplb_placement_strategy
        self.dispatch_strategy = config.moe_eplb_dispatch_strategy

        self.expert_fc1_numel = 2 * config.hidden_size * config.moe_ffn_hidden_size
        self.expert_fc2_numel = config.hidden_size * config.moe_ffn_hidden_size
        self.expert_total_numel = self.expert_fc1_numel + self.expert_fc2_numel

        self.runtime = ultra_ep.Manager(
            group=self.group,
            num_layers=config.num_layers,
            num_local_master_experts=self.num_local_master_experts,
            num_local_redundant_experts=self.num_local_redundant_experts,
            expert_fc1_numel=self.expert_fc1_numel,
            expert_fc2_numel=self.expert_fc2_numel,
            explicitly_destroy=False,
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

    def initialize_placement_random(self, seed: int = 42):
        """
        Random placement: each rank randomly selects `num_local_redundant_experts` 
        experts from OTHER ranks to replicate.
        """
        setup_placement_random(
            num_ranks=self.num_ranks,
            num_local_master=self.num_local_master_experts,
            num_local_redundant=self.num_local_redundant_experts,
            physical_to_logical_map=self.physical_to_logical_map,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_counts=self.logical_replica_counts,
            replica_distribution="uniform",
            seed=seed,
        )
        self.physical_to_logical_map_gpu = self.physical_to_logical_map.to(device='cuda')
        self.logical_to_physical_map_gpu = self.logical_to_physical_map.to(device='cuda')
        self.logical_replica_counts_gpu = self.logical_replica_counts.to(device='cuda')
    
    def reroute_random(
        self,
        layer_id: int,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand routing map from [num_tokens, num_global_logical_experts] to 
        [num_tokens, num_global_physical_experts].
        
        """
        num_tokens = routing_map.shape[0]
        device = routing_map.device
        
        # Must be created from scratch to avoid grad collapse
        expanded_routing_map = torch.zeros(
            (num_tokens, self.num_global_physical_experts),
            dtype=routing_map.dtype, device=device
        )
        expanded_probs = torch.zeros(
            (num_tokens, self.num_global_physical_experts),
            dtype=probs.dtype, device=device
        )

        ## Random dispatch
        logical_to_physical = self.logical_to_physical_map_gpu[layer_id]
        replica_counts = self.logical_replica_counts_gpu[layer_id]
        
        # Find all (token, logical_expert) pairs that are routed
        token_indices, logical_indices = routing_map.nonzero(as_tuple=True)
        
        if len(token_indices) == 0:
            return expanded_routing_map, expanded_probs
        
        # Get prob values for these pairs (maintains gradient flow)
        prob_values = probs[token_indices, logical_indices]
        
        # For each routed pair, select a physical expert from the candidates
        # Vectorized: get replica counts and candidates for each logical expert
        counts_per_token = replica_counts[logical_indices]
        
        # Generate random choices within [0, count) for each token
        random_vals = torch.rand(len(token_indices), device=device)
        random_choice = (random_vals * counts_per_token.float()).long()
        
        # Gather the physical indices: logical_to_physical[logical_indices, random_choice]
        physical_indices = logical_to_physical[logical_indices, random_choice]
        
        # Set the expanded routing map (no gradient needed for boolean mask)
        expanded_routing_map[token_indices, physical_indices] = True
        
        # For expanded_probs, use scatter to maintain gradient flow
        # Flatten to 1D for scatter operation
        flat_indices = token_indices * self.num_global_physical_experts + physical_indices
        expanded_probs_flat = expanded_probs.view(-1)
        
        # scatter is differentiable w.r.t. src (prob_values)
        expanded_probs = expanded_probs_flat.scatter(
            0, flat_indices, prob_values
        ).view(num_tokens, self.num_global_physical_experts) 
            
        return expanded_routing_map, expanded_probs


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