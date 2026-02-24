import os
import pandas as pd
import click
import numpy as np
from scipy.stats import norm
import pingouin as pg

from compositionality_study.constants import (
    MEMORY_TEST_DIR,
    BIDS_DIR
)

def calculate_dprime(hits, n_signals, fa, n_noise):
    """
    Calculates d' using the loglinear correction to handle edge cases
    (hit or false-alarm rate of 0 or 1).

    Applies the correction: rate_corrected = (count + 0.5) / (total + 1)

    Parameters
    ----------
    hits : int
        Number of correctly identified signal (real) items.
    n_signals : int
        Total number of signal (real) items.
    fa : int
        Number of noise (lure) items incorrectly identified as seen.
    n_noise : int
        Total number of noise (lure) items.

    Returns
    -------
    float
        d' value.
    """
    hr = (hits + 0.5) / (n_signals + 1)
    far = (fa + 0.5) / (n_noise + 1)
    return float(norm.ppf(hr) - norm.ppf(far))


def calculate_session_performance(response_path, ground_truth_path):
    """
    Loads and merges a single session's response and ground truth files.

    Returns a merged DataFrame with per-trial columns including ``modality``,
    ``complexity``, ``label``, ``truth_is_real``, ``user_says_seen``, and
    ``is_correct``. Returns None on any error.
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

        # 'x' means seen (Signal), empty/nan means not seen
        df_resp['user_says_seen'] = (
            df_resp['write_X_if_seen'].fillna('').astype(str).str.lower().str.strip() == 'x'
        )

        # Load Ground Truth
        if not os.path.exists(ground_truth_path):
            print(f"Warning: Ground truth file not found: {ground_truth_path}")
            return None

        df_gt = pd.read_csv(ground_truth_path)
        # 'real' means seen before (Signal), 'lure' means new (Noise)
        df_gt['truth_is_real'] = df_gt['label'] == 'real'

        # Normalise stimulus key to str: the ground truth CSV sometimes stores
        # image IDs as float (e.g. 357598.0) while the response TSV has str.
        df_gt['stimulus'] = df_gt['stimulus'].astype(str).str.strip()
        df_resp['stimulus'] = df_resp['stimulus'].astype(str).str.strip()

        # Inner join: only grade items present in both files
        merged = pd.merge(df_gt, df_resp, on='stimulus', how='inner')

        if merged.empty:
            print("Warning: No matching stimuli found between response and ground truth.")
            return None

        merged['is_correct'] = merged['user_says_seen'] == merged['truth_is_real']
        return merged

    except Exception as e:
        print(f"Error processing {os.path.basename(response_path)}: {e}")
        return None


def _fmt_mean_sd(values, decimals=2):
    """Return 'M ± SD' string for an iterable of values."""
    m = np.mean(values)
    s = np.std(values, ddof=1)
    fmt = f".{decimals}f"
    return f"{m:{fmt}} \u00b1 {s:{fmt}}"


def _anova_row_str(row):
    """Format a single pingouin rm_anova result row as a compact string.

    Supports both older pingouin (partial eta-squared as 'np2') and newer
    versions (generalised eta-squared as 'ng2').
    """
    p = row['p-unc']
    p_str = f"= {p:.3f}".lstrip('0') if p >= 0.001 else "< .001"
    if 'np2' in row.index:
        eta_val = row['np2']
        eta_label = "\u03b7\u00b2_p"
    else:
        eta_val = row['ng2']
        eta_label = "\u03b7\u00b2_G"
    return (
        f"F({int(row['ddof1'])},{int(row['ddof2'])}) = {row['F']:.2f}, "
        f"p {p_str}, {eta_label} = {eta_val:.3f}"
    ), float(p)


def _print_results_paragraph(df_conditions, df_summary, anova_dprime, anova_acc):
    """
    Print a ready-to-use results paragraph with all XX placeholders filled in.
    """
    # False alarm rate per subject
    df_summary = df_summary.copy()
    df_summary['far'] = df_summary['fa'] / df_summary['n_noise']
    far_str = _fmt_mean_sd(df_summary['far'].dropna())

    # d' means per modality, averaged over complexity within subject first
    img_dp = df_conditions[df_conditions['modality'] == 'image'].groupby('subject')['dprime'].mean()
    txt_dp = df_conditions[df_conditions['modality'] == 'text'].groupby('subject')['dprime'].mean()

    # ANOVA rows
    sources = ['modality', 'complexity', 'modality * complexity']
    dp_stats = {}
    acc_stats = {}
    for src in sources:
        row_dp = anova_dprime[anova_dprime['Source'] == src].iloc[0]
        row_acc = anova_acc[anova_acc['Source'] == src].iloc[0]
        dp_stats[src] = _anova_row_str(row_dp)
        acc_stats[src] = _anova_row_str(row_acc)

    # Build d' findings clause
    significant = [src for src in sources if dp_stats[src][1] < 0.05]
    if significant:
        clauses = []
        for src in significant:
            stat_str, _ = dp_stats[src]
            if src == 'modality':
                clauses.append(
                    f"a significant main effect of modality ({stat_str}), such that "
                    f"images (d\u2019 = {_fmt_mean_sd(img_dp)}) were better recognised "
                    f"than text (d\u2019 = {_fmt_mean_sd(txt_dp)})"
                )
            elif src == 'complexity':
                clauses.append(f"a significant main effect of complexity ({stat_str})")
            else:
                clauses.append(f"a significant modality \u00d7 complexity interaction ({stat_str})")
        dp_clause = "; ".join(clauses)
    else:
        all_p_dp = min(v[1] for v in dp_stats.values())
        mod_s, _ = dp_stats['modality']
        cmp_s, _ = dp_stats['complexity']
        iax_s, _ = dp_stats['modality * complexity']
        dp_clause = (
            f"no significant main effect of modality ({mod_s}), "
            f"complexity ({cmp_s}), or their interaction ({iax_s}) "
            f"(all ps > .{int(all_p_dp * 100):02d})"
        )

    # Build hit-rate findings clause
    sig_acc = [src for src in sources if acc_stats[src][1] < 0.05]
    if sig_acc:
        hr_parts = []
        for src in sig_acc:
            stat_str, _ = acc_stats[src]
            label = 'modality' if src == 'modality' else (
                'complexity' if src == 'complexity' else 'their interaction'
            )
            hr_parts.append(f"{label} ({stat_str})")
        hr_clause = "a significant effect of " + " and ".join(hr_parts)
    else:
        all_p_acc = min(v[1] for v in acc_stats.values())
        hr_clause = (
            f"no significant effect of modality, complexity, or their interaction "
            f"(all ps > .{int(all_p_acc * 100):02d})"
        )

    paragraph = (
        f"To further characterise recognition performance, we computed d\u2019 for each "
        f"participant as a measure of discriminability, using hit rates derived separately "
        f"for each condition (modality \u00d7 complexity) and a pooled false alarm rate "
        f"across all lure items. The mean false alarm rate was {far_str}. "
        f"A 2\u00d72 repeated-measures ANOVA on d\u2019 scores revealed {dp_clause}, "
        f"suggesting that recognition performance was broadly comparable across conditions. "
        f"A parallel ANOVA on hit rates revealed {hr_clause}."
    )

    print("\n=== Results Paragraph ===")
    print(paragraph)


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
    "--output_file", default=None, type=str, help="Optional path to save condition-level results as CSV."
)
def evaluate_memory_performance(start_subject, end_subject, n_days, bids_dir, ground_truth_dir, output_file):
    """
    Evaluates memory test performance for all subjects.

    For each subject, computes per-condition hit rate and d' across the four
    cells of the 2x2 factorial design (modality: image/text x complexity:
    high/low). Lure items (shared across conditions) provide the global false-
    alarm rate used in all d' calculations. Runs 2x2 repeated-measures ANOVAs
    via pingouin on both d' and hit rate.
    """
    MODALITIES = ['image', 'text']
    COMPLEXITIES = ['high', 'low']
    CONDITIONS = [(m, c) for m in MODALITIES for c in COMPLEXITIES]

    all_condition_rows = []  # Per-subject x condition rows for ANOVA
    subject_summary = []     # Per-subject overall stats

    for subject_id in range(start_subject, end_subject + 1):
        subject_trials = []

        for day in range(1, n_days + 1):
            sub_str = f"sub-{subject_id:02d}"
            ses_str = f"ses-{day:02d}"

            # Response: /bids/sub-XX/ses-XX/beh/sub-XX_ses-XX_task-mem_beh.tsv
            resp_file = os.path.join(
                bids_dir,
                sub_str,
                ses_str,
                'beh',
                f"{sub_str}_{ses_str}_task-mem_beh.tsv"
            )
            # Ground truth: uses integer IDs in filename (e.g. subject_2, not subject_02)
            gt_file = os.path.join(
                ground_truth_dir,
                f"sub-{subject_id}",
                f"ses-{day}",
                f"memory_test_day_{day}_subject_{subject_id}_with_labels.csv"
            )

            merged = calculate_session_performance(resp_file, gt_file)

            if merged is not None:
                merged['subject'] = subject_id
                merged['session'] = day
                subject_trials.append(merged)

        if not subject_trials:
            print(f"No valid sessions found for subject {subject_id}, skipping.")
            continue

        # Combine all sessions for this subject
        df_subj = pd.concat(subject_trials, ignore_index=True)

        # Overall accuracy (across all items and all sessions)
        overall_acc = df_subj['is_correct'].mean()

        # Global false alarms: all lure items pooled across sessions.
        # Lures share a single FA rate across all four conditions.
        df_lures = df_subj[~df_subj['truth_is_real']]
        n_noise = len(df_lures)
        fa = int(df_lures['user_says_seen'].sum())

        # Per-condition d' and hit rate
        for modality, complexity in CONDITIONS:
            df_cond = df_subj[
                df_subj['truth_is_real'] &
                (df_subj['modality'] == modality) &
                (df_subj['complexity'] == complexity)
            ]
            n_signals = len(df_cond)
            hits = int(df_cond['user_says_seen'].sum())

            if n_signals == 0:
                print(
                    f"Warning: No real items for sub-{subject_id}, "
                    f"{modality}/{complexity}. Skipping condition."
                )
                dprime = np.nan
                hit_rate = np.nan
            else:
                dprime = calculate_dprime(hits, n_signals, fa, n_noise)
                hit_rate = hits / n_signals

            all_condition_rows.append({
                'subject': subject_id,
                'modality': modality,
                'complexity': complexity,
                'n_signals': n_signals,
                'hits': hits,
                'n_noise': n_noise,
                'fa': fa,
                'dprime': dprime,
                'hit_rate': hit_rate,
            })

        subject_summary.append({
            'subject': subject_id,
            'overall_accuracy': round(overall_acc, 4),
            'total_correct': int(df_subj['is_correct'].sum()),
            'total_trials': len(df_subj),
            'fa': fa,
            'n_noise': n_noise,
        })

    if not all_condition_rows:
        print("No results calculated.")
        return

    df_conditions = pd.DataFrame(all_condition_rows)
    df_summary = pd.DataFrame(subject_summary)

    # Condition means and SDs
    df_cond_means = (
        df_conditions
        .groupby(['modality', 'complexity'])[['dprime', 'hit_rate']]
        .agg(['mean', 'std'])
    )

    print("\n=== Overall Subject Performance ===")
    print(df_summary.to_string(index=False))

    print("\n=== Per-Condition Means \u00b1 SD (d\u2019 and Hit Rate) ===")
    print(df_cond_means.round(3).to_string())

    # Keep only subjects with complete 2x2 data for ANOVA
    df_anova = df_conditions[['subject', 'modality', 'complexity', 'dprime', 'hit_rate']].dropna()
    n_per_subject = df_anova.groupby('subject').size()
    complete_subjects = n_per_subject[n_per_subject == len(CONDITIONS)].index
    df_anova = df_anova[df_anova['subject'].isin(complete_subjects)].copy()

    if df_anova['subject'].nunique() < 2:
        print("\nNot enough subjects with complete 2x2 data for ANOVA (need \u2265 2).")
        return

    print(f"\n(ANOVAs based on {df_anova['subject'].nunique()} subjects with complete 2x2 data)")

    print("\n=== 2x2 Repeated-Measures ANOVA on d\u2019 ===")
    anova_dprime = pg.rm_anova(
        data=df_anova,
        dv='dprime',
        within=['modality', 'complexity'],
        subject='subject',
        detailed=True,
        correction=False,  # type: ignore
    )
    print(anova_dprime.round(4).to_string(index=False))

    print("\n=== 2x2 Repeated-Measures ANOVA on Hit Rate ===")
    anova_acc = pg.rm_anova(
        data=df_anova,
        dv='hit_rate',
        within=['modality', 'complexity'],
        subject='subject',
        detailed=True,
        correction=False,  # type: ignore
    )
    print(anova_acc.round(4).to_string(index=False))

    _print_results_paragraph(df_conditions, df_summary, anova_dprime, anova_acc)

    if output_file:
        df_conditions.to_csv(output_file, index=False)
        summary_path = output_file.replace('.csv', '_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"\nCondition-level data saved to {output_file}")
        print(f"Subject summary saved to {summary_path}")


if __name__ == "__main__":
    evaluate_memory_performance()