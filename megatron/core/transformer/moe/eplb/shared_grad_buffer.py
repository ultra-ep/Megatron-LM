"""
EPLB Shared Gradient Buffer Module

This module provides a shared gradient buffer for EPLB redundant (replica) experts.
Instead of allocating separate main_grad buffers for each replica expert in each layer,
a single shared buffer pool is allocated and reused across all layers.

Key insight: Replica expert gradients are reduced and added to master expert gradients
layer-by-layer during backward pass. After the reduction is complete for a layer,
its replica gradient buffer is no longer needed and can be reused by other layers.

Memory savings:
- Without sharing: num_layers * num_redundant_per_rank * weight_size * 2 (fc1 + fc2)
- With sharing: num_redundant_per_rank * weight_size * 2 (fc1 + fc2)
- Savings: (num_layers - 1) * num_redundant_per_rank * weight_size * 2

Usage:
1. Create a single EPLBSharedGradBufferManager instance for all MoE layers
2. Register replica parameters from each layer
3. The manager assigns main_grad views from the shared buffer
"""

from typing import Dict, List, Optional, Tuple
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EPLBSharedGradBufferManager:
    """
    Manages a shared gradient buffer pool for EPLB replica experts across all layers.
    
    This manager allocates a single contiguous buffer that is large enough to hold
    all replica expert gradients for one layer. Since backward pass processes layers
    sequentially and replica gradients are reduced immediately, this buffer can be
    reused across all layers.

    Optionally, this manager also holds the shared buffer for global logical experts grad, 
    if all-reduce is adopted to reduce replica gradients to master experts.
    
    The buffer is organized as:
    - fc1 weights: [num_redundant_per_rank, fc1_weight_numel]
    - fc2 weights: [num_redundant_per_rank, fc2_weight_numel]
    
    Attributes:
        num_redundant_per_rank: Number of redundant experts per EP rank
        grad_dtype: Data type for gradient buffer (typically float32)
        device: Device to allocate buffer on
        fc1_weight_shape: Shape of fc1 weight tensor
        fc2_weight_shape: Shape of fc2 weight tensor
    """
    
    def __init__(
        self,
        num_redundant_per_rank: int,
        fc1_weight_shape: torch.Size,
        fc2_weight_shape: torch.Size,
        grad_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the shared gradient buffer manager.
        
        Args:
            num_redundant_per_rank: Number of redundant experts per rank
            fc1_weight_shape: Shape of a single fc1 expert weight
            fc2_weight_shape: Shape of a single fc2 expert weight
            grad_dtype: Data type for gradients (default float32 for accumulation)
            device: Device to allocate buffer on (default: current CUDA device)
        """
        self.num_redundant_per_rank = num_redundant_per_rank
        self.fc1_weight_shape = fc1_weight_shape
        self.fc2_weight_shape = fc2_weight_shape
        self.grad_dtype = grad_dtype
        self.device = device if device is not None else torch.cuda.current_device()
        
        # Calculate buffer sizes
        self.fc1_numel_per_expert = fc1_weight_shape.numel()
        self.fc2_numel_per_expert = fc2_weight_shape.numel()
        self.total_numel_per_expert = self.fc1_numel_per_expert + self.fc2_numel_per_expert
        self.fc1_total_numel = self.fc1_numel_per_expert * num_redundant_per_rank
        self.fc2_total_numel = self.fc2_numel_per_expert * num_redundant_per_rank
        self.total_numel = self.fc1_total_numel + self.fc2_total_numel
        
        # Allocate the shared buffer
        self._buffer: Optional[torch.Tensor] = None
        self._fc1_views: List[torch.Tensor] = []
        self._fc2_views: List[torch.Tensor] = []
        
        # Track registered layers for debugging
        self._registered_layers: List[str] = []
        
        # Initialize flag
        self._initialized = False

        # (Optional) Global logical experts buffer
        self._global_logical_experts_grad_buffer: Optional[torch.Tensor] = None
    
    def _allocate_buffer(self):
        """Allocate the shared gradient buffer."""
        if self._buffer is not None:
            return
            
        self._buffer = torch.zeros(
            self.total_numel,
            dtype=self.grad_dtype,
            device=self.device,
            requires_grad=False,
        )
        
        # Create views for fc1 and fc2
        fc1_buffer = self._buffer[:self.fc1_total_numel]
        fc2_buffer = self._buffer[self.fc1_total_numel:]
        
        # Create individual expert views
        for i in range(self.num_redundant_per_rank):
            fc1_start = i * self.fc1_numel_per_expert
            fc1_end = fc1_start + self.fc1_numel_per_expert
            self._fc1_views.append(
                fc1_buffer[fc1_start:fc1_end].view(self.fc1_weight_shape)
            )
            
            fc2_start = i * self.fc2_numel_per_expert
            fc2_end = fc2_start + self.fc2_numel_per_expert
            self._fc2_views.append(
                fc2_buffer[fc2_start:fc2_end].view(self.fc2_weight_shape)
            )
        
        self._initialized = True
        logger.info(
            f"EPLBSharedGradBufferManager: Allocated shared gradient buffer "
            f"with {self.total_numel:,} elements ({self.total_numel * 4 / 1024**2:.2f} MB for float32)"
        )
    
    def get_or_create_global_logical_experts_grad_buffer(self, num_global_logical: Optional[int] = None):
        """Allocate the shared gradient buffer for global logical experts."""
        if self._global_logical_experts_grad_buffer is not None:
            return self._global_logical_experts_grad_buffer
            
        assert num_global_logical is not None, "num_global_logical must be provided"
        self._global_logical_experts_grad_buffer = torch.zeros(
            (num_global_logical, self.total_numel_per_expert),
            dtype=self.grad_dtype,
            device=self.device,
        )
        return self._global_logical_experts_grad_buffer
    
    def get_fc1_grad_view(self, replica_idx: int) -> torch.Tensor:
        """
        Get a gradient buffer view for fc1 weight of a replica expert.
        
        Args:
            replica_idx: Index of the replica expert (0 to num_redundant_per_rank-1)
            
        Returns:
            Tensor view into the shared buffer with shape fc1_weight_shape
        """
        if not self._initialized:
            self._allocate_buffer()
        
        assert 0 <= replica_idx < self.num_redundant_per_rank, \
            f"replica_idx {replica_idx} out of range [0, {self.num_redundant_per_rank})"
        
        return self._fc1_views[replica_idx]
    
    def get_fc2_grad_view(self, replica_idx: int) -> torch.Tensor:
        """
        Get a gradient buffer view for fc2 weight of a replica expert.
        
        Args:
            replica_idx: Index of the replica expert (0 to num_redundant_per_rank-1)
            
        Returns:
            Tensor view into the shared buffer with shape fc2_weight_shape
        """
        if not self._initialized:
            self._allocate_buffer()
        
        assert 0 <= replica_idx < self.num_redundant_per_rank, \
            f"replica_idx {replica_idx} out of range [0, {self.num_redundant_per_rank})"
        
        return self._fc2_views[replica_idx]
    
    def assign_main_grad_to_replica_params(
        self,
        linear_fc1: nn.Module,
        linear_fc2: nn.Module,
        num_local_master: int,
        layer_name: str = "",
    ):
        """
        Assign shared gradient buffer views to replica expert parameters.
        
        This method sets the main_grad attribute of replica expert weights to
        views into the shared buffer, enabling gradient accumulation directly
        into the shared buffer.
        
        Args:
            linear_fc1: The fc1 linear module (TEGroupedLinear)
            linear_fc2: The fc2 linear module (TEGroupedLinear)
            num_local_master: Number of local master experts
            layer_name: Name of the layer for logging
        """
        if not self._initialized:
            self._allocate_buffer()
        
        for replica_idx in range(self.num_redundant_per_rank):
            local_physical_idx = num_local_master + replica_idx
            
            # Assign fc1 main_grad
            fc1_weight = getattr(linear_fc1, f'weight{local_physical_idx}', None)
            if fc1_weight is not None:
                fc1_weight.main_grad = self._fc1_views[replica_idx]
            
            # Assign fc2 main_grad
            fc2_weight = getattr(linear_fc2, f'weight{local_physical_idx}', None)
            if fc2_weight is not None:
                fc2_weight.main_grad = self._fc2_views[replica_idx]
        
        self._registered_layers.append(layer_name)
        logger.debug(f"EPLBSharedGradBufferManager: Assigned main_grad to layer '{layer_name}'")
    
    def zero_grad(self):
        """Zero out the entire shared gradient buffer.
        
        This should be called at the start of each iteration to reset gradients.
        Since all replica experts share this buffer, a single zero operation
        resets gradients for all layers.
        """
        if self._buffer is not None:
            self._buffer.zero_()
    
    def zero_fc1_grad(self):
        """Zero out only the fc1 gradient portion of the shared buffer."""
        if self._buffer is not None:
            self._buffer[:self.fc1_total_numel].zero_()
    
    def zero_fc2_grad(self):
        """Zero out only the fc2 gradient portion of the shared buffer."""
        if self._buffer is not None:
            self._buffer[self.fc1_total_numel:].zero_()
    
    def get_fc1_stacked_grads(self) -> torch.Tensor:
        """Get all fc1 replica gradients as a stacked tensor [num_redundant, *weight_shape].
        
        This returns a view into the shared buffer, allowing efficient bulk operations
        without copying data. The returned tensor has shape:
        [num_redundant_per_rank, *fc1_weight_shape]
        """
        if not self._initialized:
            self._allocate_buffer()
        
        fc1_buffer = self._buffer[:self.fc1_total_numel]
        return fc1_buffer.view(self.num_redundant_per_rank, *self.fc1_weight_shape)
    
    def get_fc2_stacked_grads(self) -> torch.Tensor:
        """Get all fc2 replica gradients as a stacked tensor [num_redundant, *weight_shape].
        
        This returns a view into the shared buffer, allowing efficient bulk operations
        without copying data. The returned tensor has shape:
        [num_redundant_per_rank, *fc2_weight_shape]
        """
        if not self._initialized:
            self._allocate_buffer()
        
        fc2_buffer = self._buffer[self.fc1_total_numel:]
        return fc2_buffer.view(self.num_redundant_per_rank, *self.fc2_weight_shape)
    
    @property
    def buffer(self) -> Optional[torch.Tensor]:
        """Get the underlying shared buffer tensor."""
        return self._buffer
    
    @property
    def memory_bytes(self) -> int:
        """Get the total memory used by the shared buffer in bytes."""
        if self._buffer is not None:
            return self._buffer.numel() * self._buffer.element_size()
        return 0
    
    def get_memory_savings_vs_independent(self, num_layers: int) -> int:
        """
        Calculate memory savings compared to independent buffers.
        
        Args:
            num_layers: Number of MoE layers
            
        Returns:
            Memory saved in bytes
        """
        independent_memory = num_layers * self.total_numel * 4  # Assume float32
        shared_memory = self.total_numel * 4
        return independent_memory - shared_memory


# Global registry for shared buffer managers
# Key: (ep_group_id, num_redundant, fc1_shape, fc2_shape)
_shared_buffer_registry: Dict[Tuple, EPLBSharedGradBufferManager] = {}


def get_or_create_shared_grad_buffer_manager(
    ep_group: torch.distributed.ProcessGroup,
    num_redundant_per_rank: int,
    fc1_weight_shape: torch.Size,
    fc2_weight_shape: torch.Size,
    grad_dtype: torch.dtype = torch.float32,
) -> EPLBSharedGradBufferManager:
    """
    Get or create a shared gradient buffer manager.
    
    This function ensures that all MoE layers with the same configuration
    share the same gradient buffer manager.
    
    Args:
        ep_group: Expert parallel process group
        num_redundant_per_rank: Number of redundant experts per rank
        fc1_weight_shape: Shape of fc1 weight
        fc2_weight_shape: Shape of fc2 weight
        grad_dtype: Gradient dtype
        
    Returns:
        EPLBSharedGradBufferManager instance
    """
    # Create a hashable key
    key = (
        id(ep_group),
        num_redundant_per_rank,
        tuple(fc1_weight_shape),
        tuple(fc2_weight_shape),
    )
    
    if key not in _shared_buffer_registry:
        manager = EPLBSharedGradBufferManager(
            num_redundant_per_rank=num_redundant_per_rank,
            fc1_weight_shape=fc1_weight_shape,
            fc2_weight_shape=fc2_weight_shape,
            grad_dtype=grad_dtype,
        )
        _shared_buffer_registry[key] = manager
        logger.info(
            f"Created new EPLBSharedGradBufferManager for {num_redundant_per_rank} replicas"
        )
    
    return _shared_buffer_registry[key]


def clear_shared_buffer_registry():
    """Clear the global shared buffer registry.
    
    This should be called when reinitializing models to avoid memory leaks.
    """
    global _shared_buffer_registry
    _shared_buffer_registry.clear()
