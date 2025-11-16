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
from typing import Dict, List, Tuple, Any, Optional

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


def load_schedule(path: str) -> Tuple[List[Tuple[float, int, int, int, float, float]], Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    schedules = data.get('schedule_results') or []
    if not schedules:
        raise ValueError('evaluate_json does not contain schedule_results')
    return schedules[-1], data


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
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    colors = {}
    for idx, stand in enumerate(sorted(set(stands_all))):
        colors[stand] = palette[idx % len(palette)]
    highlight = {stand: '#ff6666' for stand in blocked}
    return colors, highlight


def _normalize_event_list(raw: Any) -> List[Dict[str, Any]]:
    events = []
    if not raw:
        return events
    for entry in raw:
        try:
            start = float(entry.get('start'))
            end = float(entry.get('end'))
        except Exception:
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        raw_stands = entry.get('stands', [])
        if isinstance(raw_stands, str):
            stands = _parse_stands(raw_stands)
        else:
            temp = []
            for s in raw_stands:
                try:
                    sid = int(s)
                except Exception:
                    continue
                if 1 <= sid <= 28:
                    temp.append(sid)
            stands = sorted(set(temp))
        resources = sorted(set(str(r).upper() for r in entry.get('resource_types', []) if str(r).strip()))
        devices_payload = entry.get('devices') or entry.get('device_ids') or []
        devices = sorted(set(str(d).upper() for d in devices_payload if str(d).strip()))
        events.append({
            'id': entry.get('id'),
            'start': start,
            'end': end,
            'stands': stands,
            'resources': resources,
            'devices': devices
        })
    events.sort(key=lambda e: e['start'])
    return events


def _events_from_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active: Dict[Any, Dict[str, Any]] = {}
    result: List[Dict[str, Any]] = []
    for entry in history or []:
        evt_id = entry.get('event_id')
        action = entry.get('action')
        if action == 'start':
            active[evt_id] = {
                'id': evt_id,
                'start': float(entry.get('time', 0.0)),
                'stands': entry.get('stands') or [],
                'resources': entry.get('resources') or [],
                'devices': entry.get('devices') or []
            }
        elif action == 'end':
            payload = active.pop(evt_id, None)
            if not payload:
                continue
            payload['end'] = float(entry.get('time', payload['start']))
            result.append(payload)
    return _normalize_event_list(result)


def _load_external_events(spec: Optional[str]) -> List[Dict[str, Any]]:
    if not spec:
        return []
    try:
        if os.path.exists(spec):
            with open(spec, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        else:
            payload = json.loads(spec)
    except Exception as exc:
        raise ValueError(f'Failed to parse disturbance events: {exc}') from exc
    if isinstance(payload, dict):
        payload = [payload]
    return _normalize_event_list(payload)


def _load_disturbance_events(eval_payload: Dict[str, Any], extra_spec: Optional[str]) -> List[Dict[str, Any]]:
    events = []
    if 'disturbance_events' in eval_payload:
        events = _normalize_event_list(eval_payload['disturbance_events'])
    elif 'disturbance' in eval_payload:
        events = _events_from_history(eval_payload.get('disturbance', {}).get('history') or [])
    if extra_spec:
        events = _normalize_event_list(events + _load_external_events(extra_spec))
    return events


def _annotate_events(ax, events, plane_rows, view_start, view_end):
    EVENT_STYLES = {
        'stands': {'color': '#ffb3ba', 'label': 'Stand outage'},
        'resources': {'color': '#b0c9ff', 'label': 'Resource outage'},
        'devices': {'color': '#b5f5c0', 'label': 'Device outage'}
    }
    legend_seen = set()
    base_y = plane_rows * 10 + 3
    type_offsets = {'stands': 0.0, 'resources': 1.4, 'devices': 2.8}
    for evt in events:
        start = max(view_start, float(evt['start']))
        end = min(view_end, float(evt['end']))
        if end <= start:
            continue
        for key, field in (('stands', 'stands'), ('resources', 'resources'), ('devices', 'devices')):
            targets = evt.get(field) or []
            if not targets:
                continue
            style = EVENT_STYLES[key]
            label = style['label']
            ax.axvspan(start, end, color=style['color'], alpha=0.18,
                       label=None if label in legend_seen else label, zorder=0.1)
            legend_seen.add(label)
            midpoint = start + (end - start) / 2.0
            text = f"{label}: {', '.join(str(x) for x in targets)}"
            ax.text(midpoint, base_y + type_offsets[key], text, color=style['color'],
                    fontsize=8, ha='center', va='bottom',
                    rotation=0, clip_on=False)


def plot_gantt(segments_by_plane, stands_all, events, disturb_start, disturb_end, pad, out_path):
    if not segments_by_plane:
        raise ValueError('No affected planes found for the disturbance window.')
    plane_ids = sorted(segments_by_plane.keys())
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(plane_ids) + 2))
    yticks, ylabels = [], []
    blocked_stands = sorted({sid for evt in events for sid in evt.get('stands', [])})
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

    if events:
        _annotate_events(ax, events, len(plane_ids), view_start, view_end)
    ax.axvspan(disturb_start, disturb_end, color='red', alpha=0.10,
               label='Focus window', zorder=0.05)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Planes')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title('Stand usage around disturbance window')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict((lab, h) for h, lab in zip(handles, labels) if lab)
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax.set_xlim(view_start, view_end)
    ax.set_ylim(-2, len(plane_ids) * 10 + 12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot stand occupancy around a disturbance event.')
    parser.add_argument('--evaluate_json', required=True,
                        help='Path to evaluate.json containing schedule_results.')
    parser.add_argument('--disturbance_start', type=float, required=True)
    parser.add_argument('--disturbance_end', type=float, required=True)
    parser.add_argument('--stands', type=str, default='')
    parser.add_argument('--disturbance_events', type=str, default='',
                        help='Optional JSON string or file containing disturbance events; '
                             'each event may include stands/devices/resource_types.')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--window_pad', type=float, default=30.0,
                        help='Minutes before/after the disturbance window to display.')
    args = parser.parse_args()

    schedule, eval_meta = load_schedule(args.evaluate_json)
    events = _load_disturbance_events(eval_meta, args.disturbance_events or None)
    event_stands = sorted({sid for evt in events for sid in evt.get('stands', [])})
    stands = _parse_stands(args.stands) if args.stands else event_stands
    segments = build_intervals(schedule)
    affected = select_affected_planes(segments, stands, args.disturbance_start, args.disturbance_end, args.window_pad)
    plot_gantt(
        affected,
        stands or list(range(1, 29)),
        events,
        args.disturbance_start,
        args.disturbance_end,
        args.window_pad,
        args.output
    )


if __name__ == '__main__':
    main()
