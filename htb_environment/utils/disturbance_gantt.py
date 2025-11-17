"""Plot disturbance-aware stand occupancy Gantt charts.

Usage:
    python utils/disturbance_gantt.py --evaluate_json result/qmix/60_agents/my_multi_batch_run/evaluate.json \
        --disturbance_start 450 --disturbance_end 550 --stands 5-10

The script reads the last schedule in evaluate.json, extracts all occupation
intervals on stands 1-28, and highlights the subset of planes that were using
any of the blocked stands within the disturbance window. The resulting Gantt
chart helps confirm that aircraft were relocated to alternative stands during
the outage.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _parse_stands(text: str) -> List[int]:
    stands = set()
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a, b = token.split('-', 1)
            try:
                lo, hi = sorted((int(a), int(b)))
            except ValueError:
                continue
            for sid in range(lo, hi + 1):
                stands.add(sid)
        else:
            try:
                stands.add(int(token))
            except ValueError:
                continue
    return sorted(s for s in stands if 1 <= s <= 28)


def load_schedule(path: str) -> List[Tuple[float, int, int, int, float, float]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    schedules = data.get('schedule_results') or []
    if not schedules:
        raise ValueError('evaluate_json does not contain schedule_results')
    return schedules[-1]


def build_intervals(schedule, stands_filter=None):
    intervals: Dict[int, List[Tuple[float, float, int]]] = {}
    for (t, job_id, site_id, plane_id, proc_min, move_min) in schedule:
        if not (1 <= site_id <= 28):
            continue
        if proc_min <= 0:
            continue
        if stands_filter and site_id not in stands_filter:
            continue
        end_t = t + float(proc_min)
        planeseg = intervals.setdefault(plane_id, [])
        planeseg.append((t, end_t, site_id))
    for segs in intervals.values():
        segs.sort(key=lambda x: x[0])
    return intervals


def select_affected_planes(all_segments, stands, disturb_start, disturb_end, pad):
    win_start = disturb_start - pad
    win_end = disturb_end + pad
    affected = {}
    for pid, segs in all_segments.items():
        keep = False
        for (s, e, stand) in segs:
            if stands and stand not in stands:
                continue
            if not (e <= win_start or s >= win_end):
                keep = True
                break
        if keep:
            affected[pid] = segs
    if affected or not stands:
        return affected
    # fallback: planes that ever touched the stands (even outside window)
    for pid, segs in all_segments.items():
        for (_, _, stand) in segs:
            if stand in stands:
                affected[pid] = segs
                break
    return affected


def _clip_segments(segs, start, end):
    clipped = []
    for (s, e, stand) in segs:
        if e <= start or s >= end:
            continue
        clipped.append((max(s, start), min(e, end), stand))
    return clipped


def _build_stand_colors(stands_all, blocked):
    palette = list(mcolors.TABLEAU_COLORS.values()) + \
        list(mcolors.CSS4_COLORS.values())
    colors = {}
    for idx, stand in enumerate(sorted(set(stands_all))):
        colors[stand] = palette[idx % len(palette)]
    highlight = {}
    for stand in blocked:
        highlight[stand] = '#ff6666'
    return colors, highlight


def plot_gantt(segments_by_plane, stands_all, blocked_stands, disturb_start, disturb_end, pad, out_path):
    if not segments_by_plane:
        raise ValueError(
            'No affected planes found for the disturbance window.')
    plane_ids = sorted(segments_by_plane.keys())
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(plane_ids) + 2))
    yticks, ylabels = [], []
    color_map, blocked_colors = _build_stand_colors(stands_all, blocked_stands)

    view_start = disturb_start - pad
    view_end = disturb_end + pad
    for row, pid in enumerate(plane_ids):
        y = row * 10
        yticks.append(y + 4)
        ylabels.append(f'Plane {pid}')
        for (start, end, stand) in _clip_segments(segments_by_plane[pid], view_start, view_end):
            width = max(0.01, end - start)
            face = blocked_colors.get(stand, color_map.get(stand, '#888888'))
            ax.broken_barh(
                [(start, width)],
                (y, 8),
                facecolors=face,
                edgecolors='black',
                linewidth=0.5,
                label=f'Stand {stand}' if stand not in color_map else None
            )

    ax.axvspan(disturb_start, disturb_end, color='red', alpha=0.15,
               label='Disturbance window')
    for stand in blocked_stands:
        ax.axhspan(-10, len(plane_ids) * 10 + 10, xmin=0, xmax=0,
                   color=blocked_colors.get(stand, '#ff6666'), alpha=0.0,
                   label=f'Blocked stand {stand}')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Planes')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title('Stand usage around disturbance window')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.set_xlim(view_start, view_end)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot stand occupancy around a disturbance event.')
    parser.add_argument('--evaluate_json', required=True,
                        help='Path to evaluate.json containing schedule_results.')
    parser.add_argument('--disturbance_start', type=float, required=True)
    parser.add_argument('--disturbance_end', type=float, required=True)
    parser.add_argument('--stands', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--window_pad', type=float, default=30.0,
                        help='Minutes before/after the disturbance window to display.')
    args = parser.parse_args()

    stands = _parse_stands(args.stands) if args.stands else []
    schedule = load_schedule(args.evaluate_json)
    segments = build_intervals(schedule)
    affected = select_affected_planes(
        segments, stands, args.disturbance_start, args.disturbance_end, args.window_pad)
    plot_gantt(affected, stands or list(range(1, 29)), stands,
               args.disturbance_start, args.disturbance_end, args.window_pad, args.output)


if __name__ == '__main__':
    main()
