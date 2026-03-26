# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import atexit
import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from megatron.core import utils

logger = logging.getLogger(__name__)

_ENABLE_ENV = "MCORE_MOE_EXPERT_LOAD_DUMP"
_SAVE_DIR_ENV = "MCORE_MOE_EXPERT_LOAD_DUMP_DIR"
_MAX_FORWARDS_ENV = "MCORE_MOE_EXPERT_LOAD_DUMP_MAX_FORWARDS"
_DEFAULT_SAVE_DIR = "/var/log/mcore_ep_loads"


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_positive_int_env(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    parsed = int(value)
    return max(parsed, 0)


@dataclass
class _WriteTask:
    save_path: str
    host_buffer: torch.Tensor
    layer_numbers: np.ndarray
    forward_counts: np.ndarray
    ep_group_ranks: np.ndarray
    ep_rank: int
    ep_size: int
    global_rank: int
    num_global_physical_experts: int
    num_local_physical_experts: int
    capture_id: str
    capture_event: Optional[torch.cuda.Event]


class ExpertLoadRecorder:
    """Capture local expert loads with an async D2H side stream.

    Each process keeps a pinned CPU tensor with shape
    ``[num_local_moe_layers, max_forwards, num_global_physical_experts]``.
    For every forward pass we:
    1. launch a side-stream reduction ``routing_map.sum(dim=0)``
    2. enqueue a non-blocking D2H copy into the preallocated host buffer
    3. defer file-system I/O to a background thread once capture is complete

    The resulting file contains local per-rank loads only. Global expert loads are
    reconstructed offline by summing the dumps from all ranks in the same EP group.
    """

    def __init__(
        self,
        ep_group: Optional[torch.distributed.ProcessGroup],
        num_global_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.enabled = _read_bool_env(_ENABLE_ENV, default=False)
        self.max_forwards = _read_positive_int_env(_MAX_FORWARDS_ENV, default=0)
        if not self.enabled or self.max_forwards <= 0:
            self.enabled = False
            self.max_forwards = 0
            return

        self.group = ep_group
        self.global_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.ep_rank = utils.get_pg_rank(ep_group)
        self.ep_size = utils.get_pg_size(ep_group)
        if torch.distributed.is_initialized() and ep_group is not None:
            self.ep_group_ranks = tuple(torch.distributed.get_process_group_ranks(ep_group))
        else:
            self.ep_group_ranks = (self.global_rank,)

        self.num_global_physical_experts = num_global_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.save_dir = os.environ.get(_SAVE_DIR_ENV, _DEFAULT_SAVE_DIR)
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.capture_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"

        self._registered_layers: List[int] = []
        self._layer_to_slot: Dict[int, int] = {}
        self._forward_counts: List[int] = []

        self._host_buffer: Optional[torch.Tensor] = None
        self._gpu_scratch: Optional[torch.Tensor] = None
        self._last_capture_event: Optional[torch.cuda.Event] = None

        self._has_capture = False
        self._write_scheduled = False
        self._closed = False

        self._write_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"mcore-ep-load-writer-rank{self.global_rank}",
            daemon=True,
        )
        self._writer.start()
        atexit.register(self.close)

    def register_layer(self, layer_number: int) -> None:
        """Register a MoE layer before the first capture."""
        if not self.enabled or layer_number in self._registered_layers:
            return
        if self._host_buffer is not None:
            raise RuntimeError(
                "Cannot register new MoE layers after expert-load capture has started."
            )
        self._registered_layers.append(layer_number)

    @property
    def save_path(self) -> str:
        return os.path.join(
            self.save_dir,
            f"mcore_ep_loads_rank{self.global_rank}_{self.capture_id}.npz",
        )

    @property
    def partial_save_path(self) -> str:
        return os.path.join(
            self.save_dir,
            f"mcore_ep_loads_rank{self.global_rank}_{self.capture_id}_partial.npz",
        )

    def _initialize_buffers(self) -> None:
        if self._host_buffer is not None:
            return
        if not self._registered_layers:
            raise RuntimeError("No MoE layers were registered for expert-load capture.")

        self._registered_layers = sorted(self._registered_layers)
        self._layer_to_slot = {
            layer_number: slot for slot, layer_number in enumerate(self._registered_layers)
        }
        self._forward_counts = [0 for _ in self._registered_layers]

        self._host_buffer = torch.empty(
            (
                len(self._registered_layers),
                self.max_forwards,
                self.num_global_physical_experts,
            ),
            dtype=torch.int32,
            pin_memory=True,
        )
        self._gpu_scratch = torch.empty(
            (self.num_global_physical_experts,),
            dtype=torch.int32,
            device=self.device,
        )

    def capture(self, layer_number: int, routing_map: torch.Tensor) -> None:
        """Enqueue a local expert-load reduction and async D2H copy."""
        if not self.enabled or self._write_scheduled:
            return
        if not routing_map.is_cuda:
            return

        self._initialize_buffers()
        layer_slot = self._layer_to_slot[layer_number]
        forward_idx = self._forward_counts[layer_slot]
        if forward_idx >= self.max_forwards:
            return

        assert self._host_buffer is not None
        assert self._gpu_scratch is not None

        host_row = self._host_buffer[layer_slot, forward_idx]
        current_stream = torch.cuda.current_stream(device=self.device)
        self.capture_stream.wait_stream(current_stream)
        routing_map.record_stream(self.capture_stream)

        with torch.cuda.stream(self.capture_stream):
            torch.sum(routing_map, dim=0, dtype=torch.int32, out=self._gpu_scratch)
            host_row.copy_(self._gpu_scratch, non_blocking=True)
            self._last_capture_event = self.capture_stream.record_event()

        self._forward_counts[layer_slot] += 1
        self._has_capture = True

        if all(count >= self.max_forwards for count in self._forward_counts):
            self._enqueue_write(partial=False)

    def _enqueue_write(self, partial: bool) -> None:
        if not self.enabled or self._write_scheduled or not self._has_capture:
            return

        assert self._host_buffer is not None
        task = _WriteTask(
            save_path=self.partial_save_path if partial else self.save_path,
            host_buffer=self._host_buffer,
            layer_numbers=np.asarray(self._registered_layers, dtype=np.int32),
            forward_counts=np.asarray(self._forward_counts, dtype=np.int32),
            ep_group_ranks=np.asarray(self.ep_group_ranks, dtype=np.int32),
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            global_rank=self.global_rank,
            num_global_physical_experts=self.num_global_physical_experts,
            num_local_physical_experts=self.num_local_physical_experts,
            capture_id=self.capture_id,
            capture_event=self._last_capture_event,
        )
        self._write_scheduled = True
        self._write_queue.put(task)

    def _writer_loop(self) -> None:
        torch.cuda.set_device(self.device.index)
        while True:
            task = self._write_queue.get()
            if task is None:
                return

            try:
                if task.capture_event is not None:
                    task.capture_event.synchronize()

                max_forward_count = int(task.forward_counts.max()) if task.forward_counts.size else 0
                loads = task.host_buffer.numpy()[:, :max_forward_count, :]
                np.savez(
                    task.save_path,
                    dump_version=np.asarray("mcore_ep_load_dump_v1"),
                    capture_id=np.asarray(task.capture_id),
                    loads=loads,
                    layer_numbers=task.layer_numbers,
                    forward_counts=task.forward_counts,
                    ep_group_ranks=task.ep_group_ranks,
                    ep_rank=np.int32(task.ep_rank),
                    ep_size=np.int32(task.ep_size),
                    global_rank=np.int32(task.global_rank),
                    num_global_physical_experts=np.int32(task.num_global_physical_experts),
                    num_local_physical_experts=np.int32(task.num_local_physical_experts),
                )
                logger.info("Saved expert-load dump to %s", task.save_path)
            except Exception:
                logger.exception("Failed to save expert-load dump to %s", task.save_path)

    def close(self) -> None:
        if not self.enabled or self._closed:
            return

        if not self._write_scheduled and self._has_capture:
            self._enqueue_write(partial=True)

        self._write_queue.put(None)
        self._writer.join(timeout=30.0)
        self._closed = True


_expert_load_recorder_registry: Dict[int, ExpertLoadRecorder] = {}


def get_or_create_expert_load_recorder(
    ep_group: Optional[torch.distributed.ProcessGroup],
    num_global_physical_experts: int,
    num_local_physical_experts: int,
) -> Optional[ExpertLoadRecorder]:
    """Return the EP-group-scoped expert-load recorder if env-based dumping is enabled."""
    if not _read_bool_env(_ENABLE_ENV, default=False):
        return None
    if _read_positive_int_env(_MAX_FORWARDS_ENV, default=0) <= 0:
        return None

    key = id(ep_group)
    recorder = _expert_load_recorder_registry.get(key)
    if recorder is None:
        recorder = ExpertLoadRecorder(
            ep_group=ep_group,
            num_global_physical_experts=num_global_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
        _expert_load_recorder_registry[key] = recorder
    else:
        if recorder.num_global_physical_experts != num_global_physical_experts:
            raise RuntimeError(
                "Mismatched global physical expert counts in the same EP-group recorder."
            )
        if recorder.num_local_physical_experts != num_local_physical_experts:
            raise RuntimeError(
                "Mismatched local physical expert counts in the same EP-group recorder."
            )

    return recorder if recorder.enabled else None


def clear_expert_load_recorder_registry() -> None:
    """Close and clear all recorder instances."""
    global _expert_load_recorder_registry
    for recorder in _expert_load_recorder_registry.values():
        recorder.close()
    _expert_load_recorder_registry = {}
