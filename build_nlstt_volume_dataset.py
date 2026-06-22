#!/usr/bin/env python3
"""Build a three-time-point NLSTt nodule-volume dataset for growth modeling.

The source file is the STMixer/NLSTt registration-coordinate table. It contains
paired lung nodule records across NLST screening years. The exported model file
uses an ellipsoid approximation from the three world-coordinate diameters:

    V_cm3 = pi / 6 * dx_world * dy_world * dz_world / 1000

This is a CT-derived nodule volume estimate, not an official NLST tumor-volume
ground truth.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "STMixer" / "data" / "registration coordinates" / "data.csv"
OUT_DIR = ROOT / "data" / "nlstt_processed"

TARGET_N_TRAJECTORIES = 800
REQUIRED_YEARS = (0, 1, 2)

# Conservative size/volume filters. These remove clearly tiny/unstable marks and
# extreme lesions where an ellipsoid from box dimensions is less reliable.
MIN_AXIS_MM = 3.0
MAX_AXIS_MM = 80.0
MIN_VOLUME_CM3 = 0.02
MAX_VOLUME_CM3 = 20.0
MAX_ADJACENT_LOG_CHANGE = math.log(12.0)


def to_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def volume_cm3(dx: float, dy: float, dz: float) -> float:
    return math.pi / 6.0 * dx * dy * dz / 1000.0


def growth_class(v0: float, v2: float) -> str:
    ratio = v2 / v0
    if ratio < 0.8:
        return "decreasing"
    if ratio < 1.25:
        return "stable"
    if ratio < 2.0:
        return "slow_growth"
    return "rapid_growth"


def read_source() -> list[dict[str, str]]:
    with SOURCE.open(newline="") as f:
        return list(csv.DictReader(f))


def build_candidates(rows: list[dict[str, str]]) -> tuple[list[dict], dict[str, int]]:
    by_nodule: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_nodule[row["nodId"]].append(row)

    stats = Counter()
    candidates = []
    for nod_id, group in by_nodule.items():
        stats["all_nodules"] += 1
        year_map = {}
        duplicate_year = False
        for row in group:
            try:
                year = int(float(row["study_year"]))
            except ValueError:
                duplicate_year = True
                continue
            if year in year_map:
                duplicate_year = True
            year_map[year] = row

        if duplicate_year:
            stats["excluded_duplicate_or_bad_year"] += 1
            continue
        if tuple(sorted(year_map)) != REQUIRED_YEARS:
            stats["excluded_not_three_required_years"] += 1
            continue

        points = []
        bad_measurement = False
        for year in REQUIRED_YEARS:
            row = year_map[year]
            dx = to_float(row["dx_world"])
            dy = to_float(row["dy_world"])
            dz = to_float(row["dz_world"])
            if dx is None or dy is None or dz is None:
                bad_measurement = True
                break
            if min(dx, dy, dz) < MIN_AXIS_MM or max(dx, dy, dz) > MAX_AXIS_MM:
                bad_measurement = True
                break
            vol = volume_cm3(dx, dy, dz)
            if vol < MIN_VOLUME_CM3 or vol > MAX_VOLUME_CM3:
                bad_measurement = True
                break
            point = {
                "patient_id": row["pId"],
                "trajectory_id": row["nodId"],
                "series_id": row["seriesId"],
                "source_record_id": row["id"],
                "t_rel": float(year),
                "study_year": year,
                "dx_world_mm": dx,
                "dy_world_mm": dy,
                "dz_world_mm": dz,
                "V_obs_cm3": vol,
            }
            points.append(point)

        if bad_measurement:
            stats["excluded_invalid_size_or_volume"] += 1
            continue

        log_changes = [
            abs(math.log(points[1]["V_obs_cm3"] / points[0]["V_obs_cm3"])),
            abs(math.log(points[2]["V_obs_cm3"] / points[1]["V_obs_cm3"])),
        ]
        if max(log_changes) > MAX_ADJACENT_LOG_CHANGE:
            stats["excluded_extreme_adjacent_change"] += 1
            continue

        v0, v1, v2 = [p["V_obs_cm3"] for p in points]
        cls = growth_class(v0, v2)
        rel_change = (v2 - v0) / v0
        log_smoothness = abs(math.log(v2 / v1) - math.log(v1 / v0))
        median_axis = median(
            [axis for p in points for axis in (p["dx_world_mm"], p["dy_world_mm"], p["dz_world_mm"])]
        )
        candidates.append(
            {
                "trajectory_id": nod_id,
                "patient_id": points[0]["patient_id"],
                "points": points,
                "growth_class": cls,
                "v0_cm3": v0,
                "v1_cm3": v1,
                "v2_cm3": v2,
                "relative_change_t2_vs_t0": rel_change,
                "max_adjacent_log_change": max(log_changes),
                "log_smoothness": log_smoothness,
                "median_axis_mm": median_axis,
            }
        )
        stats["eligible_three_timepoint_trajectories"] += 1
    return candidates, dict(stats)


def balanced_select(candidates: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        groups[item["growth_class"]].append(item)

    # Prefer trajectories with larger measured lesions and smoother temporal
    # consistency, while keeping all growth classes represented.
    for group in groups.values():
        group.sort(key=lambda x: (-x["median_axis_mm"], x["log_smoothness"], x["trajectory_id"]))

    classes = ["decreasing", "stable", "slow_growth", "rapid_growth"]
    base = TARGET_N_TRAJECTORIES // len(classes)
    selected = []
    leftovers = []
    for cls in classes:
        group = groups.get(cls, [])
        take = min(base, len(group))
        selected.extend(group[:take])
        leftovers.extend(group[take:])

    if len(selected) < TARGET_N_TRAJECTORIES:
        leftovers.sort(key=lambda x: (-x["median_axis_mm"], x["log_smoothness"], x["trajectory_id"]))
        selected.extend(leftovers[: TARGET_N_TRAJECTORIES - len(selected)])

    selected.sort(key=lambda x: (x["patient_id"], x["trajectory_id"]))
    return selected


def write_outputs(selected: list[dict], candidates: list[dict], stats: dict[str, int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUT_DIR / "nlstt_three_scan_volume_model_input.csv"
    full_path = OUT_DIR / "nlstt_three_scan_volume_long_with_metadata.csv"
    summary_path = OUT_DIR / "nlstt_three_scan_trajectory_summary.csv"
    all_eligible_summary_path = OUT_DIR / "nlstt_three_scan_all_eligible_summary.csv"
    metadata_path = OUT_DIR / "nlstt_three_scan_selection_metadata.txt"

    model_fields = ["patient_id", "t_rel", "V_obs"]
    full_fields = [
        "patient_id",
        "trajectory_id",
        "study_year",
        "t_rel",
        "V_obs_cm3",
        "dx_world_mm",
        "dy_world_mm",
        "dz_world_mm",
        "series_id",
        "source_record_id",
        "growth_class",
    ]
    summary_fields = [
        "trajectory_id",
        "patient_id",
        "growth_class",
        "v0_cm3",
        "v1_cm3",
        "v2_cm3",
        "relative_change_t2_vs_t0",
        "max_adjacent_log_change",
        "log_smoothness",
        "median_axis_mm",
    ]

    with model_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_fields)
        writer.writeheader()
        for item in selected:
            for p in item["points"]:
                writer.writerow(
                    {
                        "patient_id": item["trajectory_id"],
                        "t_rel": f'{p["t_rel"]:.6g}',
                        "V_obs": f'{p["V_obs_cm3"]:.10g}',
                    }
                )

    with full_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_fields)
        writer.writeheader()
        for item in selected:
            for p in item["points"]:
                writer.writerow(
                    {
                        **{k: p[k] for k in full_fields if k in p},
                        "V_obs_cm3": f'{p["V_obs_cm3"]:.10g}',
                        "growth_class": item["growth_class"],
                    }
                )

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    k: (f"{item[k]:.10g}" if isinstance(item[k], float) else item[k])
                    for k in summary_fields
                }
            )

    with all_eligible_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for item in sorted(candidates, key=lambda x: (x["patient_id"], x["trajectory_id"])):
            writer.writerow(
                {
                    k: (f"{item[k]:.10g}" if isinstance(item[k], float) else item[k])
                    for k in summary_fields
                }
            )

    selected_counts = Counter(item["growth_class"] for item in selected)
    eligible_counts = Counter(item["growth_class"] for item in candidates)
    unique_patients = len({item["patient_id"] for item in selected})
    with metadata_path.open("w") as f:
        f.write("NLSTt three-scan CT-derived nodule-volume dataset\n")
        f.write("Source: STMixer/data/registration coordinates/data.csv\n")
        f.write("Volume: ellipsoid approximation pi/6 * dx_world * dy_world * dz_world / 1000 cm^3\n")
        f.write("Important: these are CT-derived lung nodule volumes, not official tumor-volume ground truth.\n\n")
        f.write("Selection criteria:\n")
        f.write(f"- Required exactly study_year {REQUIRED_YEARS} for the same nodId.\n")
        f.write(f"- Required positive finite dx/dy/dz world-coordinate diameters.\n")
        f.write(f"- Axis range: [{MIN_AXIS_MM}, {MAX_AXIS_MM}] mm.\n")
        f.write(f"- Volume range: [{MIN_VOLUME_CM3}, {MAX_VOLUME_CM3}] cm^3 at all time points.\n")
        f.write(f"- Max adjacent absolute log-volume change <= log(12).\n")
        f.write(f"- Target sample size: {TARGET_N_TRAJECTORIES}, selected by growth-class stratification.\n\n")
        f.write("Source filtering counts:\n")
        for key in sorted(stats):
            f.write(f"- {key}: {stats[key]}\n")
        f.write(f"\nEligible growth-class counts: {dict(sorted(eligible_counts.items()))}\n")
        f.write(f"Selected growth-class counts: {dict(sorted(selected_counts.items()))}\n")
        f.write(f"Selected trajectories: {len(selected)}\n")
        f.write(f"Selected unique NLST subjects: {unique_patients}\n")
        f.write(f"Rows in model input: {len(selected) * 3}\n")
        f.write("\nOutput files:\n")
        f.write(f"- {model_path.name}: minimal patient_id,t_rel,V_obs model input.\n")
        f.write(f"- {full_path.name}: long-format file with dimensions and identifiers.\n")
        f.write(f"- {summary_path.name}: one row per selected nodule trajectory.\n")
        f.write(f"- {all_eligible_summary_path.name}: one row per all eligible nodule trajectory.\n")


def main() -> None:
    rows = read_source()
    candidates, stats = build_candidates(rows)
    selected = balanced_select(candidates)
    write_outputs(selected, candidates, stats)
    print(f"Loaded rows: {len(rows)}")
    print(f"Eligible trajectories: {len(candidates)}")
    print(f"Selected trajectories: {len(selected)}")
    print(f"Selected unique patients: {len({x['patient_id'] for x in selected})}")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
