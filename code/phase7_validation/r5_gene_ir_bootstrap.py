import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '<WORK_DIR>/analysis/r5_replication'
DATA_PATH = '<WORK_DIR>/data/variant_cc_with_alphgenome.csv'
IR_PATH = '<WORK_DIR>/results/final_valid_audit_20260202/all_gene_ir.csv'

N_BOOTSTRAP = 1000
SEED = 42
THRESHOLD_PCT = 90
MIN_VARIANTS = 10
MIN_HIGH_LOW = 3

SIGNIFICANT_MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
MODALITY_LABELS = {
    'rna_seq_effect': 'RNA-seq',
    'cage_effect': 'CAGE',
    'dnase_effect': 'DNase',
    'chip_histone_effect': 'ChIP-histone'
}

print("=" * 70)
print("Prompt 6: Gene-specific IR Bootstrap Confidence Intervals")
print("Started: {}".format(datetime.now()))
print("=" * 70)

print("\n" + "=" * 70)
print("Step 1: Loading and preparing data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['total_AC'] = df['case_AC'] + df['ctrl_AC']
df = df[df['total_AC'] >= 3].copy()
df = df.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')
df['is_case_enriched'] = df['enrichment'].isin(['case_enriched', 'case_only'])
print("Variants (AC>=3, unique): {:,}".format(len(df)))
print("Genes: {}".format(df['gene_name'].nunique()))
print("Case-enriched: {:,} ({:.1f}%)".format(
    df['is_case_enriched'].sum(), df['is_case_enriched'].mean() * 100))

ir_verified = pd.read_csv(IR_PATH)
if 'gene' in ir_verified.columns and 'gene_name' not in ir_verified.columns:
    ir_verified = ir_verified.rename(columns={'gene': 'gene_name'})
print("Verified gene IR values loaded: {} genes".format(len(ir_verified)))


def calc_gene_ir(data, gene_name, modality='rna_seq_effect', threshold_pct=90):
    gene_df = data[data['gene_name'] == gene_name]
    gene_mod = gene_df[gene_df[modality].notna()]

    if len(gene_mod) < MIN_VARIANTS:
        return None

    threshold = gene_mod[modality].quantile(threshold_pct / 100)
    high = gene_mod[modality] >= threshold
    low = gene_mod[modality] < threshold

    n_high = high.sum()
    n_low = low.sum()

    if n_high < MIN_HIGH_LOW or n_low < MIN_HIGH_LOW:
        return None

    high_case_pct = gene_mod.loc[high, 'is_case_enriched'].mean()
    low_case_pct = gene_mod.loc[low, 'is_case_enriched'].mean()

    if low_case_pct == 0:
        return None

    ir = high_case_pct / low_case_pct
    return ir


def calc_gene_ir_from_subset(gene_variants, modality, threshold_pct=90):
    mod_vals = gene_variants[modality].dropna()
    if len(mod_vals) < MIN_VARIANTS:
        return np.nan

    gene_mod = gene_variants[gene_variants[modality].notna()]
    threshold = gene_mod[modality].quantile(threshold_pct / 100)

    high = gene_mod[modality] >= threshold
    low = gene_mod[modality] < threshold

    n_high = high.sum()
    n_low = low.sum()

    if n_high < MIN_HIGH_LOW or n_low < MIN_HIGH_LOW:
        return np.nan

    high_case_pct = gene_mod.loc[high, 'is_case_enriched'].mean()
    low_case_pct = gene_mod.loc[low, 'is_case_enriched'].mean()

    if low_case_pct == 0:
        return np.nan

    return high_case_pct / low_case_pct


print("\n" + "=" * 70)
print("Step 3: Bootstrap CI calculation ({} iterations)".format(N_BOOTSTRAP))
print("=" * 70)

np.random.seed(SEED)

all_results = []
genes = sorted(df['gene_name'].unique())
total_genes = len(genes)

for gi, gene in enumerate(genes):
    gene_data = df[df['gene_name'] == gene].copy()
    n_variants = len(gene_data)

    if n_variants < MIN_VARIANTS:
        continue

    for mod in SIGNIFICANT_MODALITIES:
        mod_data = gene_data[gene_data[mod].notna()]
        n_mod = len(mod_data)

        if n_mod < MIN_VARIANTS:
            continue

        point_ir = calc_gene_ir_from_subset(mod_data, mod, THRESHOLD_PCT)
        if np.isnan(point_ir):
            continue

        boot_irs = []
        for b in range(N_BOOTSTRAP):
            boot_idx = np.random.choice(mod_data.index, size=n_mod, replace=True)
            boot_sample = mod_data.loc[boot_idx]
            boot_ir = calc_gene_ir_from_subset(boot_sample, mod, THRESHOLD_PCT)
            boot_irs.append(boot_ir)

        boot_irs = np.array(boot_irs)
        valid_boots = boot_irs[~np.isnan(boot_irs)]

        if len(valid_boots) < 100:
            continue

        ci_low = np.percentile(valid_boots, 2.5)
        ci_high = np.percentile(valid_boots, 97.5)
        boot_mean = np.mean(valid_boots)
        boot_se = np.std(valid_boots)

        z0 = stats.norm.ppf(np.mean(valid_boots < point_ir))

        jack_irs = []
        for j in range(min(n_mod, 200)):
            jack_data = mod_data.drop(mod_data.index[j])
            jack_ir = calc_gene_ir_from_subset(jack_data, mod, THRESHOLD_PCT)
            jack_irs.append(jack_ir)

        jack_irs = np.array([x for x in jack_irs if not np.isnan(x)])
        if len(jack_irs) > 2:
            jack_mean = np.mean(jack_irs)
            jack_diff = jack_mean - jack_irs
            a = np.sum(jack_diff ** 3) / (6 * (np.sum(jack_diff ** 2)) ** 1.5) if np.sum(jack_diff ** 2) > 0 else 0

            alpha_low = 0.025
            alpha_high = 0.975
            z_low = stats.norm.ppf(alpha_low)
            z_high = stats.norm.ppf(alpha_high)

            denom_low = 1 - a * (z0 + z_low)
            denom_high = 1 - a * (z0 + z_high)

            if denom_low != 0 and denom_high != 0:
                adj_low = stats.norm.cdf(z0 + (z0 + z_low) / denom_low)
                adj_high = stats.norm.cdf(z0 + (z0 + z_high) / denom_high)
                adj_low = max(0.001, min(0.999, adj_low))
                adj_high = max(0.001, min(0.999, adj_high))
                bca_ci_low = np.percentile(valid_boots, adj_low * 100)
                bca_ci_high = np.percentile(valid_boots, adj_high * 100)
            else:
                bca_ci_low = ci_low
                bca_ci_high = ci_high
        else:
            bca_ci_low = ci_low
            bca_ci_high = ci_high

        ci_excludes_1 = (ci_low > 1.0) or (ci_high < 1.0)
        bca_excludes_1 = (bca_ci_low > 1.0) or (bca_ci_high < 1.0)

        if point_ir > 1.0:
            direction = 'case_enriched'
        elif point_ir < 1.0:
            direction = 'ctrl_enriched'
        else:
            direction = 'neutral'

        all_results.append({
            'gene': gene,
            'modality': mod,
            'modality_label': MODALITY_LABELS[mod],
            'n_variants': n_mod,
            'point_ir': point_ir,
            'boot_mean': boot_mean,
            'boot_se': boot_se,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'bca_ci_low': bca_ci_low,
            'bca_ci_high': bca_ci_high,
            'n_valid_boots': len(valid_boots),
            'ci_excludes_1': ci_excludes_1,
            'bca_excludes_1': bca_excludes_1,
            'direction': direction
        })

    if (gi + 1) % 10 == 0 or gi == total_genes - 1:
        print("  Processed {}/{} genes, {} results so far".format(
            gi + 1, total_genes, len(all_results)))

results_df = pd.DataFrame(all_results)
print("\nTotal gene × modality results: {}".format(len(results_df)))
print("Genes with results: {}".format(results_df['gene'].nunique()))

print("\n" + "=" * 70)
print("Step 4: Shrinkage correction")
print("=" * 70)

for mod in SIGNIFICANT_MODALITIES:
    mod_results = results_df[results_df['modality'] == mod].copy()
    if len(mod_results) < 3:
        continue

    grand_mean_ir = mod_results['point_ir'].mean()
    overall_var = mod_results['point_ir'].var()

    k = len(mod_results)
    if k <= 2:
        continue

    within_var = (mod_results['boot_se'] ** 2).mean()
    between_var = np.sum((mod_results['point_ir'].values - grand_mean_ir) ** 2)

    if between_var > 0:
        shrink_factor = max(0, 1 - (k - 2) * within_var / between_var)
    else:
        shrink_factor = 0

    shrunk_irs = grand_mean_ir + shrink_factor * (mod_results['point_ir'].values - grand_mean_ir)

    for i, idx in enumerate(mod_results.index):
        results_df.loc[idx, 'shrunk_ir'] = shrunk_irs[i]
        results_df.loc[idx, 'shrinkage_factor'] = shrink_factor

    print("  {}: grand mean IR = {:.4f}, shrinkage factor = {:.4f}".format(
        MODALITY_LABELS[mod], grand_mean_ir, shrink_factor))

print("\n" + "=" * 70)
print("Step 5: Summary statistics")
print("=" * 70)

print("\n--- Per-modality summary ---")
print("{:20s} {:>6s} {:>8s} {:>12s} {:>12s} {:>8s} {:>8s}".format(
    'Modality', 'N_genes', 'Mean_IR', 'Median_IR', 'Grand_CI', 'CI>1', 'CI<1'))
print("-" * 80)

for mod in SIGNIFICANT_MODALITIES:
    mod_res = results_df[results_df['modality'] == mod]
    n_genes = len(mod_res)
    mean_ir = mod_res['point_ir'].mean()
    median_ir = mod_res['point_ir'].median()
    grand_ci = "[{:.3f}-{:.3f}]".format(mod_res['ci_low'].median(), mod_res['ci_high'].median())
    n_above = (mod_res['ci_low'] > 1.0).sum()
    n_below = (mod_res['ci_high'] < 1.0).sum()
    print("{:20s} {:>6d} {:>8.4f} {:>12.4f} {:>12s} {:>8d} {:>8d}".format(
        MODALITY_LABELS[mod], n_genes, mean_ir, median_ir, grand_ci, n_above, n_below))

rna_results = results_df[results_df['modality'] == 'rna_seq_effect']
sig_above = rna_results[rna_results['ci_low'] > 1.0].sort_values('point_ir', ascending=False)
sig_below = rna_results[rna_results['ci_high'] < 1.0].sort_values('point_ir')

print("\n--- Genes with RNA-seq 95% CI entirely above 1.0 ({} genes) ---".format(len(sig_above)))
print("{:15s} {:>7s} {:>8s} {:>20s}".format('Gene', 'N_var', 'IR', '95% CI'))
print("-" * 55)
for _, row in sig_above.iterrows():
    print("{:15s} {:>7.0f} {:>8.3f} [{:.3f}-{:.3f}]".format(
        row['gene'], row['n_variants'], row['point_ir'], row['ci_low'], row['ci_high']))

print("\n--- Genes with RNA-seq 95% CI entirely below 1.0 ({} genes) ---".format(len(sig_below)))
print("{:15s} {:>7s} {:>8s} {:>20s}".format('Gene', 'N_var', 'IR', '95% CI'))
print("-" * 55)
for _, row in sig_below.iterrows():
    print("{:15s} {:>7.0f} {:>8.3f} [{:.3f}-{:.3f}]".format(
        row['gene'], row['n_variants'], row['point_ir'], row['ci_low'], row['ci_high']))

print("\n--- Cross-modality consistency ---")
rna_sig = set(rna_results[rna_results['bca_excludes_1']]['gene'].values)
for mod in ['cage_effect', 'dnase_effect', 'chip_histone_effect']:
    mod_res = results_df[results_df['modality'] == mod]
    mod_sig = set(mod_res[mod_res['bca_excludes_1']]['gene'].values)
    overlap = rna_sig & mod_sig
    print("  RNA-seq ∩ {}: {} genes {}".format(
        MODALITY_LABELS[mod], len(overlap),
        sorted(overlap) if len(overlap) <= 10 else '(showing top 10)'))

print("\n" + "=" * 70)
print("Step 6: Comparison with verified all_gene_ir.csv")
print("=" * 70)

rna_ir = results_df[results_df['modality'] == 'rna_seq_effect'][['gene', 'point_ir', 'ci_low', 'ci_high', 'n_variants']].copy()
rna_ir = rna_ir.rename(columns={'point_ir': 'bootstrap_ir'})

merged = rna_ir.merge(ir_verified[['gene_name', 'IR']], left_on='gene', right_on='gene_name', how='inner')
merged = merged.rename(columns={'IR': 'verified_ir'})

if len(merged) > 0:
    corr, pval = stats.spearmanr(merged['bootstrap_ir'], merged['verified_ir'])
    print("Spearman correlation (bootstrap IR vs verified IR): r={:.4f}, p={:.2e}, N={}".format(
        corr, pval, len(merged)))

    max_diff = (merged['bootstrap_ir'] - merged['verified_ir']).abs().max()
    mean_diff = (merged['bootstrap_ir'] - merged['verified_ir']).abs().mean()
    print("Max |diff|: {:.4f}, Mean |diff|: {:.4f}".format(max_diff, mean_diff))

    within_ci = ((merged['verified_ir'] >= merged['ci_low']) &
                 (merged['verified_ir'] <= merged['ci_high'])).sum()
    print("Verified IR within bootstrap 95% CI: {}/{} ({:.1f}%)".format(
        within_ci, len(merged), within_ci / len(merged) * 100))

print("\n" + "=" * 70)
print("Step 7: Generating forest plot")
print("=" * 70)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    for mod in SIGNIFICANT_MODALITIES:
        mod_res = results_df[results_df['modality'] == mod].copy()
        mod_res = mod_res.sort_values('point_ir', ascending=True).reset_index(drop=True)

        if len(mod_res) < 5:
            continue

        fig, ax = plt.subplots(figsize=(10, max(8, len(mod_res) * 0.35)))

        y_positions = range(len(mod_res))

        for i, (_, row) in enumerate(mod_res.iterrows()):
            color = '#2166ac' if row['ci_high'] < 1.0 else '#b2182b' if row['ci_low'] > 1.0 else '#666666'
            marker_size = max(3, min(8, row['n_variants'] / 50))

            ax.plot([row['ci_low'], row['ci_high']], [i, i], color=color, linewidth=1.0, alpha=0.7)
            ax.plot(row['point_ir'], i, 'o', color=color, markersize=marker_size, zorder=5)

        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(mod_res['gene'].values, fontsize=7)
        ax.set_xlabel('Interaction Ratio (IR)', fontsize=11)
        ax.set_title('Gene-specific IR with 95% Bootstrap CI\n{} (N={} genes)'.format(
            MODALITY_LABELS[mod], len(mod_res)), fontsize=12)

        above_patch = mpatches.Patch(color='#b2182b', label='CI > 1 (case-enriched)')
        below_patch = mpatches.Patch(color='#2166ac', label='CI < 1 (ctrl-enriched)')
        neutral_patch = mpatches.Patch(color='#666666', label='CI spans 1')
        ax.legend(handles=[above_patch, neutral_patch, below_patch], loc='lower right', fontsize=9)

        ax.set_xlim(max(0, mod_res['ci_low'].min() - 0.2), mod_res['ci_high'].max() + 0.2)

        plt.tight_layout()
        fig_path = '{}/prompt6_forest_{}.png'.format(OUT_DIR, mod.replace('_effect', ''))
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(fig_path))

    fig, axes = plt.subplots(1, 4, figsize=(32, max(8, results_df['gene'].nunique() * 0.35)),
                             sharey=False)

    for ax_idx, mod in enumerate(SIGNIFICANT_MODALITIES):
        ax = axes[ax_idx]
        mod_res = results_df[results_df['modality'] == mod].copy()
        mod_res = mod_res.sort_values('point_ir', ascending=True).reset_index(drop=True)

        for i, (_, row) in enumerate(mod_res.iterrows()):
            color = '#2166ac' if row['ci_high'] < 1.0 else '#b2182b' if row['ci_low'] > 1.0 else '#666666'
            marker_size = max(3, min(7, row['n_variants'] / 50))

            ax.plot([row['ci_low'], row['ci_high']], [i, i], color=color, linewidth=0.8, alpha=0.7)
            ax.plot(row['point_ir'], i, 'o', color=color, markersize=marker_size, zorder=5)

        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_yticks(range(len(mod_res)))
        ax.set_yticklabels(mod_res['gene'].values, fontsize=6)
        ax.set_xlabel('IR', fontsize=10)
        ax.set_title(MODALITY_LABELS[mod], fontsize=11, fontweight='bold')

    plt.suptitle('Gene-specific Interaction Ratios with 95% Bootstrap CI\n(1,000 iterations, {} genes)'.format(
        results_df['gene'].nunique()), fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = '{}/prompt6_forest_combined.png'.format(OUT_DIR)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: {}".format(fig_path))

    rna_res = results_df[results_df['modality'] == 'rna_seq_effect'].copy()
    if 'shrunk_ir' in rna_res.columns and rna_res['shrunk_ir'].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(rna_res['point_ir'], rna_res['shrunk_ir'], alpha=0.6,
                   s=rna_res['n_variants'] / 5, c='#2166ac', edgecolors='white', linewidth=0.5)
        lims = [min(rna_res['point_ir'].min(), rna_res['shrunk_ir'].min()) - 0.1,
                max(rna_res['point_ir'].max(), rna_res['shrunk_ir'].max()) + 0.1]
        ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.5)
        ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
        ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.3)
        ax.set_xlabel('Original IR', fontsize=11)
        ax.set_ylabel('Shrunk IR (James-Stein)', fontsize=11)
        ax.set_title('RNA-seq: Shrinkage Effect on Gene IR\n(Bubble size = N variants)', fontsize=12)
        plt.tight_layout()
        fig_path = '{}/prompt6_shrinkage_rna_seq.png'.format(OUT_DIR)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(fig_path))

except ImportError:
    print("  matplotlib not available, skipping figures")
except Exception as e:
    print("  Figure generation error: {}".format(e))

print("\n" + "=" * 70)
print("Step 8: Saving results")
print("=" * 70)

results_path = '{}/prompt6_bootstrap_IR_results.tsv'.format(OUT_DIR)
results_df.to_csv(results_path, sep='\t', index=False)
print("Saved: {} ({} rows)".format(results_path, len(results_df)))

rna_summary = results_df[results_df['modality'] == 'rna_seq_effect'][
    ['gene', 'n_variants', 'point_ir', 'boot_mean', 'boot_se',
     'ci_low', 'ci_high', 'bca_ci_low', 'bca_ci_high',
     'ci_excludes_1', 'bca_excludes_1', 'direction']
].copy()
if 'shrunk_ir' in results_df.columns:
    shrunk = results_df[results_df['modality'] == 'rna_seq_effect'][['gene', 'shrunk_ir']]
    rna_summary = rna_summary.merge(shrunk, on='gene', how='left')

rna_summary = rna_summary.sort_values('point_ir')
rna_path = '{}/prompt6_rna_seq_gene_IR_CI.tsv'.format(OUT_DIR)
rna_summary.to_csv(rna_path, sep='\t', index=False)
print("Saved: {} ({} genes)".format(rna_path, len(rna_summary)))

cross_mod = results_df.pivot_table(
    index='gene', columns='modality_label',
    values=['point_ir', 'ci_low', 'ci_high', 'ci_excludes_1'],
    aggfunc='first'
)
cross_mod.columns = ['{}_{}'.format(val, mod) for val, mod in cross_mod.columns]
cross_path = '{}/prompt6_cross_modality_IR_CI.tsv'.format(OUT_DIR)
cross_mod.to_csv(cross_path, sep='\t')
print("Saved: {} ({} genes)".format(cross_path, len(cross_mod)))

print("\n" + "=" * 70)
print("Step 9: Key findings")
print("=" * 70)

for mod in SIGNIFICANT_MODALITIES:
    mod_res = results_df[results_df['modality'] == mod]
    n_total = len(mod_res)
    n_sig_above = (mod_res['bca_ci_low'] > 1.0).sum()
    n_sig_below = (mod_res['bca_ci_high'] < 1.0).sum()
    n_spanning = n_total - n_sig_above - n_sig_below
    mean_ir = mod_res['point_ir'].mean()
    median_ir = mod_res['point_ir'].median()

    print("\n  {}: {} genes".format(MODALITY_LABELS[mod], n_total))
    print("    Mean IR: {:.4f}, Median IR: {:.4f}".format(mean_ir, median_ir))
    print("    CI > 1 (case-enriched signal): {} ({:.1f}%)".format(n_sig_above, n_sig_above/n_total*100 if n_total > 0 else 0))
    print("    CI < 1 (ctrl-enriched signal): {} ({:.1f}%)".format(n_sig_below, n_sig_below/n_total*100 if n_total > 0 else 0))
    print("    CI spans 1 (not significant): {} ({:.1f}%)".format(n_spanning, n_spanning/n_total*100 if n_total > 0 else 0))

print("\n--- Overall Summary ---")
total_gene_mod = len(results_df)
total_sig = results_df['bca_excludes_1'].sum()
print("Total gene x modality tests: {}".format(total_gene_mod))
print("Significant (BCa CI excludes 1): {} ({:.1f}%)".format(total_sig, total_sig/total_gene_mod*100 if total_gene_mod > 0 else 0))

print("\n" + "=" * 70)
print("PROMPT 6 COMPLETE")
print("=" * 70)
print("Bootstrap iterations: {}".format(N_BOOTSTRAP))
print("Genes analyzed: {}".format(results_df['gene'].nunique()))
print("Modalities: {}".format(len(SIGNIFICANT_MODALITIES)))
print("Total results: {}".format(len(results_df)))
print("Finished: {}".format(datetime.now()))
