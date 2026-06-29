# batch_smoother.py
"""
Batch processing tool to apply smoothing to all .vtp/.vtm simulation results
found in a directory. Smoothed files are saved to a 'SMOOTHED' subfolder
within the input directory.
"""
import pyvista as pv
import os
import glob
import csv
import time
from datetime import datetime
from tqdm import tqdm
import argparse

# Import the core smoothing function from our library file
from beamontarget.io.smooth_results import apply_smoothing

# --- CONFIGURATION PANEL ---
# This now acts as the DEFAULT input directory if none is provided via command line.
DEFAULT_INPUT_DIRECTORY = "OUTPUT"

# The prefix to add to the smoothed output files.
OUTPUT_PREFIX = "smoothed_"

# The smoothing radius in the same units as the mesh (typically metres).
# All cells whose centroid is within this radius of a given cell will
# contribute to that cell's smoothed power density.
SMOOTHING_RADIUS = 0.02

# Maximum cell area (m²) for smoothing.  Only cells smaller than this
# threshold will be smoothed; larger cells keep their original density.
# Set to None to smooth all cells.
MAX_CELL_AREA = 1e-6

# --- END OF CONFIGURATION ---

def batch_process_directory(input_dir, radius=None, max_cell_area=None, normal_threshold_deg=7.0):
    """
    Finds all result files in the input directory, applies smoothing,
    and saves them to a 'SMOOTHED' subfolder within that directory.
    """
    if radius is None:
        radius = SMOOTHING_RADIUS
    if max_cell_area is None:
        max_cell_area = MAX_CELL_AREA
    if not os.path.isdir(input_dir):
        print(f"FATAL ERROR: Input directory '{input_dir}' not found.")
        return
        
    # --- NEW: Dynamically create the output directory path ---
    output_dir = os.path.join(input_dir, "SMOOTHED")
    os.makedirs(output_dir, exist_ok=True)
    
    search_path_vtp = os.path.join(input_dir, '*.vtp')
    search_path_vtm = os.path.join(input_dir, '*.vtm')
    # Important: Exclude files in the SMOOTHED subfolder from being processed again
    files_to_process = sorted([f for f in glob.glob(search_path_vtp) + glob.glob(search_path_vtm) if "SMOOTHED" not in os.path.dirname(f)])
    
    if not files_to_process:
        print(f"No .vtp or .vtm files found in the root of '{input_dir}'. Nothing to do.")
        return

    print(f"--- Starting Batch Smoothing Process ---")
    print(f"Input directory:  '{input_dir}'")
    print(f"Output directory: '{output_dir}'")
    print(f"Smoothing radius: {radius}")
    print(f"Max cell area:    {max_cell_area}")
    print(f"Found {len(files_to_process)} files to process.")

    batch_t0 = time.time()
    file_stats = []  # list of (filename, stats_dict_or_None)

    for input_path in tqdm(files_to_process, desc="Smoothing Files"):
        try:
            dataset = pv.read(input_path)
            dataset_copy = dataset.copy(deep=True)
            
            if isinstance(dataset_copy, pv.MultiBlock):
                processed_dataset = pv.MultiBlock()
                combined_stats = None
                for i in range(dataset_copy.n_blocks):
                    smoothed_block, blk_stats = apply_smoothing(
                        dataset_copy[i], radius=radius, max_cell_area=max_cell_area,
                        normal_threshold_deg=normal_threshold_deg)
                    processed_dataset.append(smoothed_block)
                    if blk_stats is not None:
                        if combined_stats is None:
                            combined_stats = dict(blk_stats)
                        else:
                            combined_stats["n_cells"] += blk_stats["n_cells"]
                            combined_stats["n_smoothed"] += blk_stats["n_smoothed"]
                            combined_stats["elapsed_s"] += blk_stats["elapsed_s"]
                            combined_stats["total_power_W"] += blk_stats["total_power_W"]
                            combined_stats["peak_density_before"] = max(
                                combined_stats["peak_density_before"],
                                blk_stats["peak_density_before"])
                            combined_stats["peak_density_after"] = max(
                                combined_stats["peak_density_after"],
                                blk_stats["peak_density_after"])
                            combined_stats["min_area_m2"] = min(
                                combined_stats["min_area_m2"],
                                blk_stats["min_area_m2"])
                            combined_stats["max_area_m2"] = max(
                                combined_stats["max_area_m2"],
                                blk_stats["max_area_m2"])
                final_dataset = processed_dataset
                stats = combined_stats
            else:
                final_dataset, stats = apply_smoothing(
                    dataset_copy, radius=radius, max_cell_area=max_cell_area,
                    normal_threshold_deg=normal_threshold_deg)


            original_filename = os.path.basename(input_path)
            output_filename = f"{OUTPUT_PREFIX}{original_filename}"
            full_output_path = os.path.join(output_dir, output_filename)
            
            final_dataset.save(full_output_path, binary=True)
            file_stats.append((original_filename, stats))
            
        except Exception as e:
            print(f"\n  - An error occurred while processing '{os.path.basename(input_path)}': {e}")
            file_stats.append((os.path.basename(input_path), None))

    batch_elapsed = time.time() - batch_t0

    # --- Write smooth_log.txt ------------------------------------------------------
    log_path = os.path.join(output_dir, "smooth_log.txt")
    try:
        with open(log_path, "w") as f:
            f.write("=" * 72 + "\n")
            f.write("  SMOOTHING LOG\n")
            f.write("=" * 72 + "\n")
            f.write(f"  Date              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Input directory   : {input_dir}\n")
            f.write(f"  Output directory  : {output_dir}\n")
            f.write(f"  Smoothing radius  : {radius} m\n")
            f.write(f"  Max cell area     : {max_cell_area}\n")
            f.write(f"  Files processed   : {len(files_to_process)}\n")
            f.write(f"  Total batch time  : {batch_elapsed:.1f} s\n")
            f.write("=" * 72 + "\n\n")

            for filename, st in file_stats:
                f.write(f"--- {filename} ---\n")
                if st is None:
                    f.write("  (no data — 'Deposited_Power_W' missing or read error)\n\n")
                    continue
                f.write(f"  Total cells           : {st['n_cells']:,}\n")
                f.write(f"  Cells smoothed        : {st['n_smoothed']:,}\n")
                f.write(f"  Smoothing time        : {st['elapsed_s']:.2f} s\n")
                f.write(f"  Total power           : {st['total_power_W']:.6g} W\n")
                f.write(f"  Peak density (before) : {st['peak_density_before']:.6e} W/m²\n")
                f.write(f"  Peak density (after)  : {st['peak_density_after']:.6e} W/m²\n")
                f.write(f"  Cell area range       : [{st['min_area_m2']:.4e}, {st['max_area_m2']:.4e}] m²\n")
                f.write("\n")

        print(f"  Log written to: {log_path}")
    except Exception as e:
        print(f"  WARNING: Could not write smooth_log.txt: {e}")

    # --- Write smoothed_summary.csv ------------------------------------------------
    summary_path = os.path.join(output_dir, "smoothed_summary.csv")
    try:
        # Collect all species suffixes across all files
        all_species = []
        for _, st in file_stats:
            if st is not None:
                for sp in sorted(st.get("species", {}).keys()):
                    if sp not in all_species:
                        all_species.append(sp)

        header = ["filename", "total_deposited_power_W", "peak_power_density_W_m2"]
        for sp in all_species:
            header.append(f"total_deposited_power_W_{sp}")
            header.append(f"peak_power_density_W_m2_{sp}")

        with open(summary_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for filename, st in file_stats:
                if st is not None:
                    row = [
                        filename,
                        f"{st['total_power_W']:.4e}",
                        f"{st['peak_density_after']:.4e}",
                    ]
                    for sp in all_species:
                        sp_st = st.get("species", {}).get(sp)
                        if sp_st:
                            row.append(f"{sp_st['total_power_W']:.4e}")
                            row.append(f"{sp_st['peak_density_after']:.4e}")
                        else:
                            row.append("N/A")
                            row.append("N/A")
                    writer.writerow(row)
                else:
                    writer.writerow([filename] + ["N/A"] * (len(header) - 1))
        print(f"  Summary CSV written to: {summary_path}")
    except Exception as e:
        print(f"  WARNING: Could not write smoothed_summary.csv: {e}")

    print("\n--- Batch smoothing process finished. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch smooth VTP/VTM simulation results.")
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"Directory containing the input files. Defaults to '{DEFAULT_INPUT_DIRECTORY}'."
    )
    parser.add_argument(
        '-r', '--radius',
        type=float,
        default=SMOOTHING_RADIUS,
        help=f"Smoothing radius in mesh units (metres). Defaults to {SMOOTHING_RADIUS}."
    )
    parser.add_argument(
        '-a', '--max_cell_area',
        type=float,
        default=MAX_CELL_AREA,
        help=f"Only smooth cells with area below this threshold (m²). Defaults to {MAX_CELL_AREA}. Set to 0 to smooth all cells."
    )
    args = parser.parse_args()
    
    area_arg = args.max_cell_area if args.max_cell_area > 0 else None
    batch_process_directory(args.input_dir, radius=args.radius, max_cell_area=area_arg)
