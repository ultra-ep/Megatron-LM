# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Expert Parallel Load Balancing (EPLB) Module

This package provides support for redundant experts in Megatron-LM for better online
expert load balance.

Key features:
- Redundant expert placement and routing map expansion
- Shared gradient buffer for replica experts (memory optimization)
- Gradient aggregation utilities for replica experts

Usage for shared gradient buffers:
    model = build_model(...)  # Build model with EPLB enabled
    model = DDP(model, ...)   # Wrap with DDP (allocates main_grad for masters)
    initialize_eplb_shared_grad_buffers(model)  # Initialize shared buffers for replicas
"""

from megatron.core.transformer.moe.eplb.manager import (
    EPLBManager,
    GlobalExpertPlacement,
    HAVE_EPLB,
    initialize_eplb_shared_grad_buffers,
)
from megatron.core.transformer.moe.eplb.shared_grad_buffer import (
    EPLBSharedGradBufferManager,
    get_or_create_shared_grad_buffer_manager,
    clear_shared_buffer_registry,
)

__all__ = [
    'EPLBManager',
    'GlobalExpertPlacement',
    'HAVE_EPLB',
    'initialize_eplb_shared_grad_buffers',
    'EPLBSharedGradBufferManager',
    'get_or_create_shared_grad_buffer_manager',
    'clear_shared_buffer_registry',
]
