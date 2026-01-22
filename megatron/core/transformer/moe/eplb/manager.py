# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Expert Parallel Load Balancing (EPLB) Module

This module provides support for redundant experts in Megatron-LM for better online 
expert load balance. Each EP rank can hold a predefined number of redundant experts 
(replicas) in device memory. These replicas are related to master experts from 
other ranks for load balancing purposes.

Key components:
- EPLBManager: Manages redundant expert placement and routing map expansion
- Gradient aggregation utilities for replica experts
- Weight synchronization utilities between masters and replicas
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from megatron.core import utils
from megatron.core.transformer.transformer_config import TransformerConfig

HAVE_EPLB = True # Placeholder for future online EPLB implementation as C++ libraries


@dataclass
class GlobalExpertPlacement:
    """Describes the placement of global experts across all ranks.
    
    Attributes:
        physical_to_logical_map: [num_global_physical_experts] 
            mapping from physical to logical expert indices
        logical_to_physical_map: [num_global_logical_experts, max_replicas]
            mapping from logical to physical expert indices, padded with -1.
            The first entry is always the master, followed by replicas.
        logical_replica_counts: [num_global_logical_experts] 
            number of replicas for each logical expert (includes master)
    
    Example:
        Suppose EP2 (2 GPUs) and 4 logical experts 0~3, each EP rank has 1 redundant expert
        - Master assignment: rank0 masters [2, 1], rank1 masters [0, 3]
        - num_local_physical_experts = 2 + 1 = 3 per rank
        - Physical layout:
          * Rank 0: physical [0,1,2] = [master(2), master(1), redundant]
          * Rank 1: physical [3,4,5] = [master(0), master(3), redundant]
        - Suppose rank0 replicates expert 3, rank1 replicates expert 1
        - physical_to_logical_map: [2, 1, 3, 0, 3, 1]
          (physical 0→2, 1→1, 2→3, 3→0, 4→3, 5→1)
        - logical_to_physical_map: [[3, -1], [1, 5], [0, -1], [4, 2]]
          (expert 0: master at phys 3; expert 1: master at phys 1, replica at phys 5;
           expert 2: master at phys 0; expert 3: master at phys 4, replica at phys 2)
        - logical_replica_counts: [1, 2, 1, 2]
    """

    physical_to_logical_map: torch.Tensor
    logical_to_physical_map: torch.Tensor
    logical_replica_counts: torch.Tensor


class EPLBManager:
    """
    Manages Expert Parallel Load Balancing with redundant experts.
    
    Responsibilities:
    1. Determine and manage redundant expert placement
    2. Expand routing map to include replica assignments
    3. Provide dispatch strategy for token-to-replica assignment
    4. Track which experts are masters vs replicas for gradient handling
    
    The manager supports different placement and dispatch strategies:
    - Placement: 'random' (current), 'balanced', 'online' (future)
    - Dispatch: 'random' (current), 'load_aware', 'online' (future)
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        num_local_master_experts: int,
        local_master_expert_indices: List[int],
        ep_group: torch.distributed.ProcessGroup,
    ):
        """
        Initialize the EPLB Manager.
        
        Args:
            config: TransformerConfig with EPLB settings
            num_local_master_experts: Number of local experts on this rank
            local_master_expert_indices: Global indices of local master experts
            ep_group: Expert parallel process group
        """
        self.config = config
        self.num_local_master_experts = num_local_master_experts
        self.local_master_expert_indices = local_master_expert_indices
        self.ep_group = ep_group
        self.ep_size = utils.get_pg_size(ep_group)
        self.ep_rank = utils.get_pg_rank(ep_group)
        
        self.num_redundant_per_rank = config.moe_num_redundant_experts_per_rank
        self.num_global_logical_experts = config.num_moe_experts
        self.placement_strategy = config.moe_eplb_placement_strategy
        self.dispatch_strategy = config.moe_eplb_dispatch_strategy
        
        # Total "physical" experts = master + all redundant across all ranks
        self.num_global_physical_experts = (
            self.num_global_logical_experts + self.ep_size * self.num_redundant_per_rank
        )
        
        # Number of experts this rank computes (local masters + local replicas)
        self.num_local_physical_experts = self.num_local_master_experts + self.num_redundant_per_rank
        self.local_physical_expert_indices = [
            self.ep_rank * self.num_local_physical_experts + i 
                for i in range(self.num_local_physical_experts)
        ]
        
        # Placement
        self.placement: Optional[GlobalExpertPlacement] = None
        self._placement_initialized = False
        
    def initialize_placement(self, seed: int = 42):
        """
        Initialize redundant expert placement.
        
        For strawman 'random' strategy: randomly assign which experts to replicate
        on each rank, avoiding replicating local experts on the same rank.
        
        Args:
            seed: Random seed for deterministic placement across ranks
        """
        if self._placement_initialized:
            return
            
        device = torch.cuda.current_device()
        
        if self.placement_strategy == "random":
            self.placement = self._random_placement(seed, device)
        elif self.placement_strategy == "balanced":
            # Future: implement balanced placement based on historical load
            # For now, fallback to random
            self.placement = self._random_placement(seed, device)
        elif self.placement_strategy == "online":
            # Future: online placement - start with random, update dynamically
            self.placement = self._random_placement(seed, device)
        else:
            raise ValueError(f"Unknown placement strategy: {self.placement_strategy}")
        
        self._placement_initialized = True
        
    def _random_placement(self, seed: int, device) -> GlobalExpertPlacement:
        """
        Random placement: each rank randomly selects `num_redundant_per_rank` 
        experts from OTHER ranks to replicate.
        
        Uses deterministic seeding so all ranks compute the same global placement.
        
        Physical expert indexing:
        - Physical index = rank * num_local_physical_experts + local_physical_idx
        - For each rank: local [0, num_master) are masters, 
                         local [num_master, num_local_physical) are redundants
        
        Args:
            seed: Random seed for reproducibility
            device: Device to create tensors on
            
        Returns:
            GlobalExpertPlacement with physical-logical mapping information
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        
        # Step 1: Gather which logical experts each rank's masters handle
        # Use actual local_master_expert_indices from all ranks via all-gather
        
        # Local master assignment tensor
        local_master_assignment = torch.tensor(
            self.local_master_expert_indices, dtype=torch.long, device=device
        )
        
        # All-gather to get global master assignment: [ep_size, num_local_master_experts]
        gathered_list = [
            torch.empty(self.num_local_master_experts, dtype=torch.long, device=device)
            for _ in range(self.ep_size)
        ]
        dist.all_gather(gathered_list, local_master_assignment, group=self.ep_group)
        global_master_assignment = torch.stack(gathered_list, dim=0)
        
        # Step 2: Each rank randomly selects logical experts to replicate (from non-local)
        # [ep_size, num_redundant_per_rank] - which logical experts each rank replicates
        global_redundant_assignment = torch.zeros(
            (self.ep_size, self.num_redundant_per_rank),
            dtype=torch.long, device=device
        )
        
        # Create a set of all logical expert indices for candidate selection
        all_logical_experts = torch.arange(
            self.num_global_logical_experts, device=device, dtype=torch.long
        )
        
        for rank in range(self.ep_size):
            # Get this rank's local master logical experts
            local_masters = global_master_assignment[rank]
            
            # Create mask for candidates (all experts not local to this rank)
            is_candidate = torch.ones(
                self.num_global_logical_experts, dtype=torch.bool, device=device
            )
            is_candidate[local_masters] = False
            
            # Get candidate experts (non-local)
            candidate_experts = all_logical_experts[is_candidate]
            
            # Randomly select which to replicate
            perm = torch.randperm(len(candidate_experts), generator=gen, device=device)
            selected = candidate_experts[perm[:self.num_redundant_per_rank]]
            global_redundant_assignment[rank] = selected.sort()[0]  # Sort for consistency
        
        # Step 3: Build physical_to_logical_map [num_global_physical_experts]
        physical_to_logical_map = torch.zeros(
            self.num_global_physical_experts, dtype=torch.long, device=device
        )
        
        for rank in range(self.ep_size):
            physical_base = rank * self.num_local_physical_experts
            # Master slots
            for local_idx in range(self.num_local_master_experts):
                physical_idx = physical_base + local_idx
                physical_to_logical_map[physical_idx] = global_master_assignment[rank, local_idx]
            # Redundant slots
            for local_idx in range(self.num_redundant_per_rank):
                physical_idx = physical_base + self.num_local_master_experts + local_idx
                physical_to_logical_map[physical_idx] = global_redundant_assignment[rank, local_idx]
        
        # Step 4: Build logical_to_physical_map [num_global_logical_experts, ep_size]
        # and logical_replica_counts [num_global_logical_experts]
        # Max replicas per expert = ep_size (master + at most ep_size-1 redundants)
        logical_to_physical_map = torch.full(
            (self.num_global_logical_experts, self.ep_size),
            fill_value=-1, dtype=torch.long, device=device
        )
        logical_replica_counts = torch.zeros(
            self.num_global_logical_experts, dtype=torch.long, device=device
        )
        
        # First pass: add master experts
        for rank in range(self.ep_size):
            physical_base = rank * self.num_local_physical_experts
            for local_idx in range(self.num_local_master_experts):
                physical_idx = physical_base + local_idx
                logical_idx = global_master_assignment[rank, local_idx].item()
                count = logical_replica_counts[logical_idx].item()
                logical_to_physical_map[logical_idx, count] = physical_idx
                logical_replica_counts[logical_idx] += 1
        
        # Second pass: add redundant experts
        for rank in range(self.ep_size):
            physical_base = rank * self.num_local_physical_experts
            for local_idx in range(self.num_redundant_per_rank):
                physical_idx = physical_base + self.num_local_master_experts + local_idx
                logical_idx = global_redundant_assignment[rank, local_idx].item()
                count = logical_replica_counts[logical_idx].item()
                if count < self.ep_size:  # Safety check
                    logical_to_physical_map[logical_idx, count] = physical_idx
                    logical_replica_counts[logical_idx] += 1
        
        return GlobalExpertPlacement(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_counts=logical_replica_counts,
        )
    
    def update_placement(self, token_stats: torch.Tensor):
        """
        Interface for online EPLB: update placement based on observed token statistics.
        
        Args:
            token_stats: [num_global_logical_experts] tensor of token counts per expert
            
        This is a placeholder for future online EPLB implementation.
        """
        # Future: implement online placement update algorithm
        # This would involve:
        # 1. Analyzing token distribution to find overloaded experts
        # 2. Deciding which experts need more replicas
        # 3. Updating placement and synchronizing weights
        pass
    
    def expand_routing_map(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand routing map from [num_tokens, num_global_logical_experts] to 
        [num_tokens, num_global_logical_experts + num_total_redundant].
        
        For tokens assigned to experts that have replicas, dispatch
        to either the master or one of the replicas based on strategy.
        
        Args:
            routing_map: [num_tokens, num_global_logical_experts] boolean mask
            probs: [num_tokens, num_global_logical_experts] routing probabilities
            
        Returns:
            expanded_routing_map: [num_tokens, num_global_physical_experts]
            expanded_probs: [num_tokens, num_global_physical_experts]
        """
        if not self._placement_initialized:
            self.initialize_placement()
            
        num_tokens = routing_map.shape[0]
        device = routing_map.device
        
        # Create expanded tensors
        expanded_routing_map = torch.zeros(
            (num_tokens, self.num_global_physical_experts),
            dtype=routing_map.dtype, device=device
        )
        expanded_probs = torch.zeros(
            (num_tokens, self.num_global_physical_experts),
            dtype=probs.dtype, device=device
        )
        
        if self.dispatch_strategy == "random":
            expanded_routing_map, expanded_probs = self._random_dispatch(
                routing_map, probs, expanded_routing_map, expanded_probs
            )
        elif self.dispatch_strategy == "load_aware":
            # Future: implement load-aware dispatch
            expanded_routing_map, expanded_probs = self._random_dispatch(
                routing_map, probs, expanded_routing_map, expanded_probs
            )
        elif self.dispatch_strategy == "online":
            # Future: implement online dispatch with real-time load info
            expanded_routing_map, expanded_probs = self._random_dispatch(
                routing_map, probs, expanded_routing_map, expanded_probs
            )
        else:
            raise ValueError(f"Unknown dispatch strategy: {self.dispatch_strategy}")
            
        return expanded_routing_map, expanded_probs
    
    def _random_dispatch(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        expanded_routing_map: torch.Tensor,
        expanded_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random dispatch: for each (token, expert) pair, if expert has replicas,
        randomly choose to send to masters or one of the replicas with equal probability.
        
        This method preserves gradient flow from probs to expanded_probs using
        differentiable scatter operations.
        
        Args:
            routing_map: [num_tokens, num_global_logical_experts] boolean routing map
            probs: [num_tokens, num_global_logical_experts] routing probabilities (requires grad)
            expanded_routing_map: Output [num_tokens, num_global_physical_experts] map
            expanded_probs: Output [num_tokens, num_global_physical_experts] probs
            
        Returns:
            Tuple of (expanded_routing_map, expanded_probs) where expanded_probs
            maintains gradient flow from the input probs
        """
        device = routing_map.device
        num_tokens = routing_map.shape[0]
        
        # Get lookup tables from placement
        logical_to_physical = self.placement.logical_to_physical_map
        replica_counts = self.placement.logical_replica_counts
        
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
    
    def get_master_expert_mask(self) -> torch.Tensor:
        """
        Get a boolean mask indicating which compute experts are masters (not replicas).
        
        Returns:
            Boolean tensor of shape [num_local_physical_experts], True for master experts
        """
        device = torch.cuda.current_device()
        mask = torch.zeros(self.num_local_physical_experts, dtype=torch.bool, device=device)
        mask[:self.num_local_master_experts] = True
        return mask


def initialize_eplb_shared_grad_buffers(model: Union[torch.nn.Module, List[torch.nn.Module]]):
    """
    Initialize shared gradient buffers for all EPLB-enabled MoE layers in the model.
    
    This function traverses the model and calls initialize_eplb_shared_grad_buffer()
    on each MoE layer that has EPLB enabled. It should be called after model construction
    is complete and after the DDP wrapper has allocated main_grad buffers.
    
    Usage:
        model = build_model(...)  # Build model
        model = DDP(model, ...)   # Wrap with DDP
        initialize_eplb_shared_grad_buffers(model)  # Initialize shared buffers
    
    Args:
        model: The model (or DDP-wrapped model, or list of model chunks) containing MoE layers
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer
        
    # Handle list of models (e.g., interleaved pipeline parallelism)
    if isinstance(model, list):
        for model_chunk in model:
            initialize_eplb_shared_grad_buffers(model_chunk)
        return
    
    # Handle DDP-wrapped models
    if hasattr(model, 'module'):
        model = model.module
    
    # Traverse all modules and find MoE layers
    for name, module in model.named_modules():
        if isinstance(module, MoELayer) and module.eplb_enabled:
            layer_name = f"layer_{module.layer_number}" if module.layer_number else ""
            module.experts.initialize_shared_grad_buffer(layer_name=layer_name)