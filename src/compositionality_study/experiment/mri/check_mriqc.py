"""Script to check MRIQC outputs for excessive framewise displacement."""
import os
import glob

import click
import pandas as pd

from compositionality_study.constants import MRIQC_DIR

@click.command()
@click.option('--mriqc-dir', default=MRIQC_DIR, help='Path to the MRIQC output directory containing sub-XX folders.')
@click.option('--fd-threshold', default=1.0, help='Framewise Displacement threshold in mm.')
@click.option('--percent-threshold', default=20.0, help='Max allowed percentage of frames exceeding FD threshold.')
def check_mriqc_outputs(mriqc_dir, fd_threshold=1.0, percent_threshold=20.0):
    """
    Checks MRIQC TSV output files for excessive framewise displacement.
    """
    print(f"Checking MRIQC outputs in: {mriqc_dir}")
    print(f"Criteria: > {percent_threshold}% of frames with FD > {fd_threshold}mm")

    # Pattern to find the timeseries TSV files shown in the screenshot
    # Check specifically for files ending in _timeseries.tsv in func folders
    # The structure in screenshot is sub-XX/ses-XX/func/sub-XX_ses-XX_task-comp_run-XX_timeseries.tsv
    search_pattern = os.path.join(mriqc_dir, 'sub-*', 'ses-*', 'func', '*_timeseries.tsv')
    files = glob.glob(search_pattern)

    if not files:
        print("No timeseries TSV files found. Please check the directory path.")
        return

    problematic_runs = []

    for tsv_file in sorted(files):
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            
            if 'framewise_displacement' not in df.columns:
                print(f"Skipping {os.path.basename(tsv_file)}: 'framewise_displacement' column not found.")
                continue

            # FD is often NaN for the first volume, fill with 0
            fd_series = df['framewise_displacement'].fillna(0)
            
            total_frames = len(fd_series)
            bad_frames = (fd_series > fd_threshold).sum()
            percent_bad = (bad_frames / total_frames) * 100

            if percent_bad > percent_threshold:
                run_info = os.path.basename(tsv_file).replace('_timeseries.tsv', '')
                problematic_runs.append({
                    'file': run_info,
                    'percent_bad': percent_bad,
                    'total_frames': total_frames,
                    'bad_frames': bad_frames
                })
                print(f"[FAIL] {run_info}: {percent_bad:.2f}% frames > {fd_threshold}mm")
            else:
                # Optional: print passed runs or keep silent
                # print(f"[PASS] {os.path.basename(tsv_file)}")
                pass

        except Exception as e:
            print(f"Error processing {tsv_file}: {e}")

    print("\n" + "="*40)
    print(f"Summary: Found {len(problematic_runs)} problematic runs.")
    for p in problematic_runs:
        print(f"  - {p['file']}: {p['percent_bad']:.2f}% bad frames ({p['bad_frames']}/{p['total_frames']})")

if __name__ == "__main__":
    check_mriqc_outputs()
