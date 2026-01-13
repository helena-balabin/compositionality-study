import os
import pandas as pd
import click
import numpy as np
import sys
from scipy.stats import binomtest

from compositionality_study.constants import (
    MEMORY_TEST_DIR,
    BIDS_DIR
)

def calculate_session_performance(response_path, ground_truth_path):
    """
    Calculates accuracy for a single session by merging response and ground truth.
    Returns (num_correct, num_total) tuple or None.
    """
    try:
        # Load Subject Response (BIDS)
        if not os.path.exists(response_path):
            print(f"Warning: Response file not found: {response_path}")
            return None
        
        # BIDS behavioral files are TSV
        df_resp = pd.read_csv(response_path, sep='\t')
        
        if df_resp.empty:
            print(f"Warning: Empty response file: {response_path}")
            return None

        # Clean up response column (x/X/nan)
        # 'x' means seen (Signal), empty/nan means not seen
        df_resp['user_says_seen'] = df_resp['write_X_if_seen'].fillna('').astype(str).str.lower().str.strip() == 'x'

        # Load Ground Truth
        if not os.path.exists(ground_truth_path):
            print(f"Warning: Ground truth file not found: {ground_truth_path}")
            return None
            
        df_gt = pd.read_csv(ground_truth_path)
        
        # 'real' means seen before (Signal), 'lure' means new (Noise)
        df_gt['truth_is_real'] = df_gt['label'] == 'real'

        # Merge on stimulus to handle order differences
        # Using inner join ensures we only grade items present in both (ignoring extra headers if any)
        merged = pd.merge(df_gt, df_resp, on='stimulus', how='inner')

        if merged.empty:
            print("Warning: No matching stimuli found between response and ground truth.")
            return None

        # Calculate Accuracy
        # Correct if (User Says Seen AND Truth is Real) OR (User Says Not Seen AND Truth is Lure)
        merged['is_correct'] = merged['user_says_seen'] == merged['truth_is_real']
        
        correct = merged['is_correct'].sum()
        total = len(merged)
        
        return correct, total

    except Exception as e:
        print(f"Error processing {os.path.basename(response_path)}: {e}")
        return None


@click.command()
@click.option("--start_subject", type=int, default=2, help="Start subject ID (default: 2).")
@click.option("--end_subject", type=int, default=9, help="End subject ID (default: 9).")
@click.option("--n_days", type=int, default=3, help="Number of days/sessions.")
@click.option(
    "--bids_dir", default=BIDS_DIR, type=str, help="Root BIDS directory containing responses."
)
@click.option(
    "--ground_truth_dir", default=MEMORY_TEST_DIR, type=str, help="Directory containing generated memory tests with labels."
)
@click.option(
    "--output_file", default=None, type=str, help="Optional path to save dataset as CSV."
)
def evaluate_memory_performance(start_subject, end_subject, n_days, bids_dir, ground_truth_dir, output_file):
    """
    Evaluates memory test performance for all subjects.
    Outputs a DataFrame with Mean and STD accuracy across sessions.
    """
    results = []

    for subject_id in range(start_subject, end_subject + 1):
        subject_accuracies = []
        subject_total_correct = 0
        subject_total_trials = 0
        
        for day in range(1, n_days + 1):
            sub_str = f"sub-{subject_id:02d}"
            ses_str = f"ses-{day:02d}"
            day_str = str(day) # Filenames use 'day_1', folders use 'ses-01'
            
            # Construct paths
            # Response: /bids/sub-01/ses-01/beh/sub-01_ses-01_task-mem_beh.tsv
            resp_file = os.path.join(
                bids_dir, 
                sub_str, 
                ses_str, 
                'beh', 
                f"{sub_str}_{ses_str}_task-mem_beh.tsv"
            )

            # Ground Truth: .../sub-1/ses-1/memory_test_day_1_subject_1_with_labels.csv
            # Note: memory_test.py output usually uses integers in filename "subject_1" not "subject_01"
            gt_file = os.path.join(
                ground_truth_dir,
                f"sub-{subject_id}",
                f"ses-{day}",
                f"memory_test_day_{day}_subject_{subject_id}_with_labels.csv"
            )

            res = calculate_session_performance(resp_file, gt_file)
            
            if res is not None:
                correct, total = res
                acc = correct / total if total > 0 else np.nan
                subject_accuracies.append(acc)
                subject_total_correct += correct
                subject_total_trials += total
            else:
                subject_accuracies.append(np.nan)
        
        # Calculate subject statistics
        valid_accs = [a for a in subject_accuracies if not np.isnan(a)]
        
        if valid_accs:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs, ddof=1) if len(valid_accs) > 1 else 0.0
            
            # Calculate p-value (Binomial test: is accuracy > 0.5?)
            try:
                # alternative='greater' tests if p > 0.5
                p_val_res = binomtest(subject_total_correct, subject_total_trials, p=0.5, alternative='greater')
                p_value = p_val_res.pvalue
            except AttributeError:
                # Fallback for older scipy versions
                from scipy.stats import binom_test
                p_value = binom_test(subject_total_correct, subject_total_trials, p=0.5, alternative='greater')

            results.append({
                'subject': subject_id,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'valid_sessions': len(valid_accs),
                'total_correct': subject_total_correct,
                'total_trials': subject_total_trials,
                'p_value': p_value
            })
        else:
             results.append({
                'subject': subject_id,
                'mean_accuracy': np.nan,
                'std_accuracy': np.nan,
                'valid_sessions': 0,
                'total_correct': 0,
                'total_trials': 0,
                'p_value': np.nan
            })

    # Create DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("No results calculated.")
        return

    # Add Summary Row
    summary_means = df_results.mean(numeric_only=True)
    summary_row = pd.DataFrame([summary_means])
    summary_row['subject'] = 'Mean' # Or 'Summary'
    
    # Concatenate using pd.concat instead of append
    df_final = pd.concat([df_results, summary_row], ignore_index=True)

    print("\n=== Memory Test Performance Summary ===")
    # Format p-value for readability
    formatters = {'p_value': lambda x: f"{x:.4g}"}
    print(df_final.to_string(index=False, formatters=formatters))
    
    if output_file:
        df_final.to_csv(output_file, index=False)
        print(f"\nSummary saved to {output_file}")

if __name__ == "__main__":
    evaluate_memory_performance()