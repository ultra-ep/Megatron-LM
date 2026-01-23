import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class P2PReplicaGradCommMap:
    """Describes the communication mapping for P2P replica gradient reduction.
    
    All attributes are GPU tensors or metadata derived from placement.
    """
    replica_master_ranks: Optional[torch.Tensor]
    replica_logical_indices: Optional[torch.Tensor]
    master_sender_ranks: Optional[torch.Tensor]
    master_logical_indices: Optional[torch.Tensor]
    master_valid_mask: Optional[torch.Tensor]
    master_recv_counts: Optional[torch.Tensor]