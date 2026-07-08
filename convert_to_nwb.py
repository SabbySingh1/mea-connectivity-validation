#!/usr/bin/env python3
"""
Convert MEA spike data (.npz) to NWB (Neurodata Without Borders) format.

Supports:
  - sim_cdkl5 / sim_burst (spycon format: spkt_s, spkid, unit_ids, loc_x, loc_y)
  - Real CDKL5 organoid recordings (same format)
  - Dale VAR simulation (spikes array: states x bins x neurons)

Output: one .nwb file per dataset, readable by any NWB-compatible tool.

Usage:
  python convert_to_nwb.py --input data/sim_hdmea_burst_spikes.npz \
                            --type spycon \
                            --session_id sim_burst \
                            --output data/sim_burst.nwb

  python convert_to_nwb.py --input /path/to/dale_spikes.npz \
                            --type dale \
                            --session_id dale_var \
                            --output data/dale_var.nwb

Install dependencies:
  pip install pynwb
"""
import argparse
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description="Convert MEA .npz spike data to NWB format")
    parser.add_argument("--input",      required=True, help="Path to input .npz spike file")
    parser.add_argument("--output",     required=True, help="Path to output .nwb file")
    parser.add_argument("--type",       required=True, choices=["spycon", "dale", "real"],
                        help="Data format: 'spycon' for sim_cdkl5/sim_burst, 'dale' for Dale VAR, 'real' for real organoid recordings")
    parser.add_argument("--session_id", default="mea_session", help="Session identifier")
    parser.add_argument("--subject",    default="in_vitro", help="Subject/prep description")
    parser.add_argument("--condition",  default="unknown", help="Drug condition or genotype (e.g. CDKL5, WT, psychedelic_name)")
    parser.add_argument("--dale_state", type=int, default=0, help="Dale VAR state index to use (default: 0)")
    parser.add_argument("--dt",         type=float, default=0.010, help="Dale VAR bin size in seconds (default: 0.010)")
    parser.add_argument("--min_rate",   type=float, default=0.5, help="Min firing rate (Hz) to include a unit (default: 0.5)")
    return parser


def load_spycon(path, min_rate):
    """Load spycon-format spike data (sim_cdkl5, sim_burst, real organoid)."""
    d = np.load(path, allow_pickle=True)
    times_s = d["spkt_s"].astype(float)
    ids     = d["spkid"].astype(int)
    order   = np.argsort(times_s)
    times_s, ids = times_s[order], ids[order]

    duration  = float(times_s[-1] - times_s[0])
    t0        = float(times_s[0])
    all_nodes = np.unique(ids)
    valid     = np.array([n for n in all_nodes if np.sum(ids == n) / duration >= min_rate])

    # Electrode positions if available
    loc_x = d["loc_x"] if "loc_x" in d else np.zeros(len(valid))
    loc_y = d["loc_y"] if "loc_y" in d else np.zeros(len(valid))

    spike_times_per_unit = {}
    for n in valid:
        mask = ids == n
        spike_times_per_unit[int(n)] = times_s[mask] - t0

    return spike_times_per_unit, valid, loc_x, loc_y, duration


def load_dale(path, state, dt):
    """Load Dale VAR spike data (states x bins x neurons)."""
    d = np.load(path, allow_pickle=True)
    spikes = d["spikes"][state]  # (T, N)
    T, N   = spikes.shape
    duration = T * dt

    spike_times_per_unit = {}
    valid = np.arange(N)
    for n in range(N):
        bins = np.where(spikes[:, n] > 0)[0]
        ts = []
        for b in bins:
            count = int(spikes[b, n])
            ts.extend([b * dt + k * dt / max(count, 1) for k in range(count)])
        spike_times_per_unit[n] = np.array(ts)

    loc_x = np.zeros(N)
    loc_y = np.zeros(N)
    return spike_times_per_unit, valid, loc_x, loc_y, duration


def convert(args):
    try:
        from pynwb import NWBFile, NWBHDF5IO
        from pynwb.device import Device
        from pynwb.ecephys import ElectrodeGroup
        from pynwb.misc import Units
        import pandas as pd
    except ImportError:
        print("ERROR: pynwb not installed. Run: pip install pynwb")
        return

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {args.input} (type={args.type})...", flush=True)
    if args.type in ["spycon", "real"]:
        spike_times, valid_nodes, loc_x, loc_y, duration = load_spycon(args.input, args.min_rate)
    elif args.type == "dale":
        spike_times, valid_nodes, loc_x, loc_y, duration = load_dale(args.input, args.dale_state, args.dt)

    N = len(valid_nodes)
    print(f"Units: {N}, Duration: {duration:.1f}s", flush=True)

    # ── Create NWB file ────────────────────────────────────────────────────────
    nwb = NWBFile(
        session_description=f"MEA recording — {args.condition}",
        identifier=args.session_id,
        session_start_time=datetime.now(tzlocal()),
        experimenter=["Sabadnoor Singh"],
        lab="Ben-Shalom Lab",
        institution="UC Berkeley",
        experiment_description=(
            "HD-MEA spike train connectivity inference benchmarking. "
            f"Condition: {args.condition}. Data type: {args.type}."
        ),
        keywords=["MEA", "connectivity", "spike trains", "CDKL5", "organoid"],
    )

    # ── Device + electrode group ───────────────────────────────────────────────
    device = nwb.create_device(name="HD-MEA", description="High-density microelectrode array", manufacturer="Maxwell Biosystems")
    electrode_group = nwb.create_electrode_group(
        name="electrodes",
        description="MEA electrode array",
        location="in vitro",
        device=device,
    )

    # ── Electrodes table ───────────────────────────────────────────────────────
    for i, node in enumerate(valid_nodes):
        x = float(loc_x[i]) if i < len(loc_x) else 0.0
        y = float(loc_y[i]) if i < len(loc_y) else 0.0
        nwb.add_electrode(
            x=x, y=y, z=0.0,
            imp=-1.0,
            location="in vitro",
            filtering="none",
            group=electrode_group,
            label=f"unit_{node}",
        )

    # ── Units table ────────────────────────────────────────────────────────────
    nwb.add_unit_column(name="unit_id", description="Original unit ID from MEA recording")
    nwb.add_unit_column(name="firing_rate_hz", description="Mean firing rate in Hz")
    nwb.add_unit_column(name="n_spikes", description="Total spike count")
    nwb.add_unit_column(name="electrode_x", description="Electrode X position (um)")
    nwb.add_unit_column(name="electrode_y", description="Electrode Y position (um)")

    for i, node in enumerate(valid_nodes):
        st = spike_times[int(node)]
        fr = len(st) / duration if duration > 0 else 0.0
        x  = float(loc_x[i]) if i < len(loc_x) else 0.0
        y  = float(loc_y[i]) if i < len(loc_y) else 0.0
        nwb.add_unit(
            spike_times=st,
            unit_id=int(node),
            firing_rate_hz=float(fr),
            n_spikes=int(len(st)),
            electrode_x=x,
            electrode_y=y,
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(out_path), "w") as io:
        io.write(nwb)

    print(f"Saved NWB file: {out_path}", flush=True)
    print(f"  Units: {N}", flush=True)
    print(f"  Duration: {duration:.1f}s", flush=True)
    print(f"  Total spikes: {sum(len(v) for v in spike_times.values())}", flush=True)


if __name__ == "__main__":
    args = get_parser().parse_args()
    convert(args)
