import torch
from typing import List, Optional
from megatron.core.transformer.moe.eplb.types import P2PReplicaGradCommMap

@torch.compile
def compute_p2p_maps_for_replica_grad_comm(
    ep_rank,
    num_local_master,
    num_redundant,
    num_local_physical,
    physical_to_logical,
    logical_to_physical,
    logical_replica_counts,
):
    """Compiled GPU-only logic for P2P map computation."""
    device = physical_to_logical.device
    
    # Replica -> Master
    replica_master_ranks = None
    replica_logical_indices = None
    if num_redundant > 0:
        replica_local_indices = torch.arange(num_redundant, device=device)
        replica_global_physical = (ep_rank * num_local_physical + 
                                    num_local_master + replica_local_indices)
        replica_logical_indices = physical_to_logical[replica_global_physical]
        master_physical_indices = logical_to_physical[replica_logical_indices, 0]
        replica_master_ranks = master_physical_indices // num_local_physical
        
    # Master <- Replicas
    master_sender_ranks = None
    master_logical_indices = None
    master_valid_mask = None
    master_recv_counts = None
    if num_local_master > 0:
        master_local_indices = torch.arange(num_local_master, device=device)
        master_global_physical = ep_rank * num_local_physical + master_local_indices
        master_logical_indices = physical_to_logical[master_global_physical]
        master_replica_counts = logical_replica_counts[master_logical_indices]
        max_replicas = logical_to_physical.shape[1]
        all_replica_physical = logical_to_physical[master_logical_indices, 1:max_replicas]
        master_sender_ranks = all_replica_physical // num_local_physical
        num_actual_replicas = master_replica_counts - 1
        col_indices = torch.arange(max_replicas - 1, device=device).unsqueeze(0)
        master_valid_mask = (col_indices < num_actual_replicas.unsqueeze(1)) & (all_replica_physical != -1)
        master_recv_counts = master_valid_mask.sum(dim=1).long()
        
    return P2PReplicaGradCommMap(
        replica_master_ranks=replica_master_ranks,
        replica_logical_indices=replica_logical_indices,
        master_sender_ranks=master_sender_ranks,
        master_logical_indices=master_logical_indices,
        master_valid_mask=master_valid_mask,
        master_recv_counts=master_recv_counts
    )

@torch.compile
def sum_and_add_replica_grads_to_master(
    stacked_recv: torch.Tensor,
    recv_counts: torch.Tensor,
    master_grads_fc1: List[Optional[torch.Tensor]],
    master_grads_fc2: List[Optional[torch.Tensor]],
    fc1_numel: int,
    fc1_shape: torch.Size,
    fc2_shape: torch.Size,
):
    """Fused kernel to sum replica gradients and add to master expert gradients.
    """
    num_local_master, max_senders, _ = stacked_recv.shape
    device = stacked_recv.device
    
    # Vectorized masked sum: [num_local_master, total_numel]
    col_indices = torch.arange(max_senders, device=device).unsqueeze(0)
    valid_mask = col_indices < recv_counts.unsqueeze(1)
    
    # Sum across senders for each master
    summed_all = (stacked_recv * valid_mask.unsqueeze(-1).to(stacked_recv.dtype)).sum(dim=1)
    
    # Split into fc1 and fc2 portions
    fc1_contribs = summed_all[:, :fc1_numel]
    fc2_contribs = summed_all[:, fc1_numel:]
    
    # Add to master gradients
    for i in range(num_local_master):
        if recv_counts[i] > 0:
            # fc1
            grad1 = master_grads_fc1[i]
            if grad1 is not None:
                grad1.add_(fc1_contribs[i].view(fc1_shape).to(grad1.dtype))
            # fc2
            grad2 = master_grads_fc2[i]
            if grad2 is not None:
                grad2.add_(fc2_contribs[i].view(fc2_shape).to(grad2.dtype))
