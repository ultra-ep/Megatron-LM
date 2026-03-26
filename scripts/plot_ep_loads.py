#!/usr/bin/env python3

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RankDump:
    path: Path
    group_ranks: Tuple[int, ...]
    ep_rank: int
    ep_size: int
    global_rank: int
    num_global_physical_experts: int
    num_local_physical_experts: int
    layer_numbers: np.ndarray
    forward_counts: np.ndarray
    loads: np.ndarray

    @property
    def layer_to_slot(self) -> Dict[int, int]:
        return {int(layer): idx for idx, layer in enumerate(self.layer_numbers.tolist())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-rank MoE expert-load dumps and render EP-group plots."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the .npz dumps produced by MCORE_MOE_EXPERT_LOAD_DUMP.",
    )
    parser.add_argument(
        "--ep-group",
        type=str,
        default=None,
        help="EP group selector. Use a 0-based discovered-group index or comma-separated global ranks.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layer numbers to plot. Defaults to all layers available in the selected EP group.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ep_load_plots"),
        help="Directory where the per-layer figures will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for saved figures.",
    )
    return parser.parse_args()


def load_rank_dump(path: Path) -> RankDump:
    with np.load(path, allow_pickle=False) as data:
        return RankDump(
            path=path,
            group_ranks=tuple(int(x) for x in np.asarray(data["ep_group_ranks"]).tolist()),
            ep_rank=int(np.asarray(data["ep_rank"]).item()),
            ep_size=int(np.asarray(data["ep_size"]).item()),
            global_rank=int(np.asarray(data["global_rank"]).item()),
            num_global_physical_experts=int(
                np.asarray(data["num_global_physical_experts"]).item()
            ),
            num_local_physical_experts=int(
                np.asarray(data["num_local_physical_experts"]).item()
            ),
            layer_numbers=np.asarray(data["layer_numbers"], dtype=np.int32),
            forward_counts=np.asarray(data["forward_counts"], dtype=np.int32),
            loads=np.asarray(data["loads"], dtype=np.int32),
        )


def discover_rank_dumps(input_dir: Path) -> Dict[Tuple[int, ...], List[RankDump]]:
    dump_paths = sorted(input_dir.rglob("mcore_ep_loads_rank*.npz"))
    if not dump_paths:
        raise FileNotFoundError(f"No .npz dump files found under {input_dir}")

    groups: Dict[Tuple[int, ...], List[RankDump]] = {}
    for path in dump_paths:
        rank_dump = load_rank_dump(path)
        groups.setdefault(rank_dump.group_ranks, []).append(rank_dump)

    for group_ranks, dumps in groups.items():
        dumps.sort(key=lambda item: item.ep_rank)
        expected_ep_size = dumps[0].ep_size
        expected_num_global_experts = dumps[0].num_global_physical_experts
        expected_num_local_experts = dumps[0].num_local_physical_experts
        seen_global_ranks = set()
        for dump in dumps:
            if dump.global_rank in seen_global_ranks:
                raise ValueError(
                    "Found multiple dump files for global rank "
                    f"{dump.global_rank} in EP group [{format_group(group_ranks)}]. "
                    "Please point --input-dir to a clean dump directory for a single run."
                )
            seen_global_ranks.add(dump.global_rank)
            if dump.ep_size != expected_ep_size:
                raise ValueError(
                    f"Inconsistent ep_size metadata inside EP group [{format_group(group_ranks)}]."
                )
            if dump.num_global_physical_experts != expected_num_global_experts:
                raise ValueError(
                    "Inconsistent global physical expert count inside EP group "
                    f"[{format_group(group_ranks)}]."
                )
            if dump.num_local_physical_experts != expected_num_local_experts:
                raise ValueError(
                    "Inconsistent local physical expert count inside EP group "
                    f"[{format_group(group_ranks)}]."
                )
        if len(dumps) != expected_ep_size:
            raise ValueError(
                f"EP group [{format_group(group_ranks)}] is missing rank dumps: "
                f"found {len(dumps)}, expected {expected_ep_size}."
            )
    return groups


def format_group(group_ranks: Sequence[int]) -> str:
    return ",".join(str(rank) for rank in group_ranks)


def format_group_summary(group_ranks: Sequence[int]) -> str:
    if not group_ranks:
        return "[]"

    expected = list(range(group_ranks[0], group_ranks[0] + len(group_ranks)))
    if list(group_ranks) == expected:
        return f"{group_ranks[0]}-{group_ranks[-1]}"

    if len(group_ranks) <= 8:
        return format_group(group_ranks)

    return f"{group_ranks[0]},{group_ranks[1]},...,{group_ranks[-2]},{group_ranks[-1]}"


def select_group(
    groups: Dict[Tuple[int, ...], List[RankDump]], selector: Optional[str]
) -> List[Tuple[int, Tuple[int, ...]]]:
    available = sorted(groups)
    if selector is None:
        return [(idx, group) for idx, group in enumerate(available)]

    selector = selector.strip()
    if selector.isdigit():
        index = int(selector)
        if index < 0 or index >= len(available):
            raise ValueError(
                f"EP group index {index} is out of range. Valid indices: 0..{len(available) - 1}"
            )
        return [(index, available[index])]

    explicit = tuple(int(part.strip()) for part in selector.split(",") if part.strip())
    if explicit not in groups:
        formatted = ", ".join(
            f"{idx}: ranks {format_group_summary(group)}" for idx, group in enumerate(available)
        )
        raise ValueError(f"Unknown EP group [{selector}]. Available groups: {formatted}")
    return [(available.index(explicit), explicit)]


def resolve_layers(dumps: Sequence[RankDump], requested_layers: Optional[Sequence[int]]) -> List[int]:
    common_layers = set(int(layer) for layer in dumps[0].layer_numbers.tolist())
    for dump in dumps[1:]:
        common_layers &= set(int(layer) for layer in dump.layer_numbers.tolist())

    if not common_layers:
        raise ValueError("The selected EP group does not share any common local layers.")

    if requested_layers is None:
        return sorted(common_layers)

    missing = sorted(set(requested_layers) - common_layers)
    if missing:
        raise ValueError(
            f"Requested layers {missing} are not available in every rank dump of the EP group."
        )
    return sorted(set(requested_layers))


def aggregate_layer(dumps: Sequence[RankDump], layer_number: int) -> Tuple[np.ndarray, np.ndarray]:
    per_rank_loads = []
    min_forward_count = None

    for dump in dumps:
        slot = dump.layer_to_slot[layer_number]
        forward_count = int(dump.forward_counts[slot])
        min_forward_count = (
            forward_count
            if min_forward_count is None
            else min(min_forward_count, forward_count)
        )
        per_rank_loads.append(dump.loads[slot])

    assert min_forward_count is not None
    if min_forward_count <= 0:
        raise ValueError(f"Layer {layer_number} has no captured forwards in the selected EP group.")

    trimmed = [rank_load[:min_forward_count] for rank_load in per_rank_loads]
    per_rank = np.stack(trimmed, axis=0)
    global_loads = per_rank.sum(axis=0)
    return per_rank, global_loads


def compute_rank_loads(global_loads: np.ndarray, dumps: Sequence[RankDump]) -> np.ndarray:
    num_local_physical_experts = dumps[0].num_local_physical_experts
    ep_size = dumps[0].ep_size
    expected_num_experts = num_local_physical_experts * ep_size
    if expected_num_experts != global_loads.shape[1]:
        raise ValueError(
            "Global physical expert count is inconsistent with EP-size/local-expert metadata: "
            f"{global_loads.shape[1]} vs {expected_num_experts}"
        )

    return global_loads.reshape(
        global_loads.shape[0], ep_size, num_local_physical_experts
    ).sum(axis=2)


def make_output_path(output_dir: Path, group_index: int, layer_number: int) -> Path:
    return output_dir / f"ep_group_{group_index}_layer_{layer_number}.png"


def plot_layer(
    dumps: Sequence[RankDump],
    group_index: int,
    group_ranks: Sequence[int],
    layer_number: int,
    output_path: Path,
    dpi: int,
) -> None:
    _, global_loads = aggregate_layer(dumps, layer_number)
    rank_loads = compute_rank_loads(global_loads, dumps)

    x = np.arange(global_loads.shape[0], dtype=np.int32)
    total_load = global_loads.sum(axis=1, keepdims=True)
    expert_shares = np.divide(
        global_loads,
        total_load,
        out=np.zeros_like(global_loads, dtype=np.float64),
        where=total_load > 0,
    ) * 100.0

    expert_max = global_loads.max(axis=1)
    expert_mean = global_loads.mean(axis=1)
    rank_max = rank_loads.max(axis=1)
    rank_mean = rank_loads.mean(axis=1)
    expert_max_over_mean = np.divide(
        expert_max,
        expert_mean,
        out=np.ones_like(expert_max, dtype=np.float64),
        where=expert_mean > 0,
    )
    rank_max_over_mean = np.divide(
        rank_max,
        rank_mean,
        out=np.ones_like(rank_max, dtype=np.float64),
        where=rank_mean > 0,
    )

    fig = plt.figure(figsize=(16, 10.8))
    grid = fig.add_gridspec(3, 1, height_ratios=[3.4, 1.35, 1.35], hspace=0.26)
    ax_stack = fig.add_subplot(grid[0])
    ax_expert = fig.add_subplot(grid[1], sharex=ax_stack)
    ax_rank = fig.add_subplot(grid[2], sharex=ax_stack)

    tab10 = plt.get_cmap("tab10")
    colors = [tab10(idx % 10) for idx in range(global_loads.shape[1])]
    ax_stack.stackplot(x, expert_shares.T, colors=colors, linewidth=0.0)
    ax_stack.set_ylabel("Share (%)")
    ax_stack.set_ylim(0.0, 100.0)
    ax_stack.set_title(
        "EP group "
        f"{group_index} (global ranks {format_group_summary(group_ranks)}) "
        f"layer {layer_number} physical expert load distribution",
        pad=14,
    )
    ax_stack.grid(alpha=0.2, linewidth=0.5)

    ax_expert.plot(x, expert_max_over_mean, color="#d62728", linewidth=1.8)
    ax_expert.set_ylabel("Ratio")
    ax_expert.set_title("Per-expert max / mean", pad=10)
    ax_expert.grid(alpha=0.25, linewidth=0.5)

    ax_rank.plot(x, rank_max_over_mean, color="#ff7f0e", linewidth=1.8)
    ax_rank.set_ylabel("Ratio")
    ax_rank.set_xlabel("Forward idx")
    ax_rank.set_title("Per-rank max / mean", pad=10)
    ax_rank.grid(alpha=0.25, linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        args = parse_args()
        groups = discover_rank_dumps(args.input_dir)
        selected_groups = select_group(groups, args.ep_group)

        if args.ep_group is None:
            print(f"Discovered {len(selected_groups)} EP groups. Exporting all of them.")

        for selected_group_index, selected_group in selected_groups:
            selected_dumps = groups[selected_group]
            layers = resolve_layers(selected_dumps, args.layers)

            print(
                "Selected EP group "
                f"{selected_group_index} with global ranks {format_group_summary(selected_group)} "
                f"and {len(selected_dumps)} rank dumps."
            )
            print(f"Plotting layers: {', '.join(str(layer) for layer in layers)}")

            for layer_number in layers:
                output_path = make_output_path(args.output_dir, selected_group_index, layer_number)
                plot_layer(
                    dumps=selected_dumps,
                    group_index=selected_group_index,
                    group_ranks=selected_group,
                    layer_number=layer_number,
                    output_path=output_path,
                    dpi=args.dpi,
                )
                print(f"Saved {output_path}")

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
