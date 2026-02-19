import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '<WORK_DIR>/analysis/additional_analyses'
np.random.seed(42)
N_BOOT = 1000
MIN_VARIANTS = 10

print("=" * 70)
print("Prompt 6: Gene-specific IR Bootstrap Confidence Intervals")
print("Started:", datetime.now())
print("=" * 70)

MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

print("\n--- Loading R4 data ---")
df = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
df['total_AC'] = df['case_AC'] + df['ctrl_AC']
df = df[df['total_AC'] >= 3].sort_values('total_AC', ascending=False) \
       .drop_duplicates('variant_id', keep='first').copy()
df['is_case_enriched'] = df['enrichment'].isin(['case_enriched', 'case_only'])
print("R4 unique variants: {:,}".format(len(df)))

thresholds = {}
for mod in MODALITIES:
    thresholds[mod] = df[mod].quantile(0.80)
    print("  {} threshold (80th pct): {:.4f}".format(mod, thresholds[mod]))


print("\n\n--- 6-1. R4 Gene-specific Bootstrap (1000 iterations) ---")

def bootstrap_gene_ir(gene_data, mod, global_threshold, n_boot=1000):
    n = len(gene_data)
    if n < MIN_VARIANTS:
        return None

    effects = gene_data[mod].values
    ce = gene_data['is_case_enriched'].values

    boot_irs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_effects = effects[idx]
        boot_ce = ce[idx]

        high_mask = boot_effects >= global_threshold
        low_mask = ~high_mask

        n_high = high_mask.sum()
        n_low = low_mask.sum()

        if n_high == 0 or n_low == 0:
            continue

        ce_high = boot_ce[high_mask].mean()
        ce_low = boot_ce[low_mask].mean()

        if ce_low > 0:
            boot_irs.append(ce_high / ce_low)
        elif ce_high > 0:
            boot_irs.append(np.nan)
        else:
            boot_irs.append(1.0)

    boot_irs = [x for x in boot_irs if not np.isnan(x) and np.isfinite(x)]

    if len(boot_irs) < 100:
        return None

    return {
        'mean': np.mean(boot_irs),
        'median': np.median(boot_irs),
        'ci_lower': np.percentile(boot_irs, 2.5),
        'ci_upper': np.percentile(boot_irs, 97.5),
        'includes_1': np.percentile(boot_irs, 2.5) <= 1.0 <= np.percentile(boot_irs, 97.5),
        'n_valid': len(boot_irs)
    }

r4_results = []
genes = df['gene_name'].value_counts()
eligible_genes = genes[genes >= MIN_VARIANTS].index.tolist()
print("Eligible genes (>= {} variants): {}".format(MIN_VARIANTS, len(eligible_genes)))

for i, gene in enumerate(eligible_genes):
    gdf = df[df['gene_name'] == gene]
    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        result = bootstrap_gene_ir(gdf, mod, thresholds[mod], N_BOOT)
        if result is None:
            continue

        high = gdf[gdf[mod] >= thresholds[mod]]
        low = gdf[gdf[mod] < thresholds[mod]]
        if len(high) > 0 and len(low) > 0:
            orig_ir = high['is_case_enriched'].mean() / low['is_case_enriched'].mean() \
                if low['is_case_enriched'].mean() > 0 else float('inf')
        else:
            orig_ir = np.nan

        r4_results.append({
            'gene': gene,
            'modality': mod_name,
            'n_variants': len(gdf),
            'IR_original': orig_ir,
            'IR_mean': result['mean'],
            'IR_median': result['median'],
            'CI_lower': result['ci_lower'],
            'CI_upper': result['ci_upper'],
            'includes_1': result['includes_1'],
            'n_valid_boots': result['n_valid']
        })

    if (i + 1) % 20 == 0:
        print("  Processed {}/{} genes...".format(i + 1, len(eligible_genes)))

r4_boot = pd.DataFrame(r4_results)
r4_boot.to_csv('{}/prompt6_gene_IR_bootstrap_R4.tsv'.format(OUT), sep='\t', index=False)
print("\nR4 bootstrap complete: {} gene×modality results".format(len(r4_boot)))

rna_r4 = r4_boot[r4_boot['modality'] == 'rna_seq']
print("\nR4 RNA-seq summary:")
print("  Total genes: {}".format(len(rna_r4)))
print("  CI excludes 1 (significant): {}".format((~rna_r4['includes_1']).sum()))
print("  IR > 1 (median): {}".format((rna_r4['IR_median'] > 1).sum()))
print("  IR < 1 (median): {}".format((rna_r4['IR_median'] < 1).sum()))

print("\n  Top 10 genes (highest IR):")
for _, row in rna_r4.nlargest(10, 'IR_median').iterrows():
    sig = '*' if not row['includes_1'] else ''
    print("    {:12s} IR={:.3f} [{:.3f}-{:.3f}] N={:.0f} {}".format(
        row['gene'], row['IR_median'], row['CI_lower'], row['CI_upper'],
        row['n_variants'], sig))

print("\n  Bottom 10 genes (lowest IR):")
for _, row in rna_r4.nsmallest(10, 'IR_median').iterrows():
    sig = '*' if not row['includes_1'] else ''
    print("    {:12s} IR={:.3f} [{:.3f}-{:.3f}] N={:.0f} {}".format(
        row['gene'], row['IR_median'], row['CI_lower'], row['CI_upper'],
        row['n_variants'], sig))


print("\n\n--- 6-2. R5 Gene-specific Bootstrap ---")

shared_file = '<WORK_DIR>/analysis/r5_replication/Phase_B_shared_variants.tsv'
shared = pd.read_csv(shared_file, sep='\t')
if 'R5_is_CE' not in shared.columns:
    if 'R5_enrichment' in shared.columns:
        shared['R5_is_CE'] = shared['R5_enrichment'].isin(['case_enriched', 'case_only'])
    elif 'is_case_enriched' in shared.columns:
        shared['R5_is_CE'] = shared['is_case_enriched']
shared['is_case_enriched'] = shared['R5_is_CE']

print("R5 shared variants: {:,}".format(len(shared)))

r5_results = []
r5_genes = shared['gene_name'].value_counts()
r5_eligible = r5_genes[r5_genes >= MIN_VARIANTS].index.tolist()
print("R5 eligible genes: {}".format(len(r5_eligible)))

for i, gene in enumerate(r5_eligible):
    gdf = shared[shared['gene_name'] == gene]
    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        if mod not in gdf.columns:
            continue
        result = bootstrap_gene_ir(gdf, mod, thresholds[mod], N_BOOT)
        if result is None:
            continue

        high = gdf[gdf[mod] >= thresholds[mod]]
        low = gdf[gdf[mod] < thresholds[mod]]
        if len(high) > 0 and len(low) > 0:
            orig_ir = high['is_case_enriched'].mean() / low['is_case_enriched'].mean() \
                if low['is_case_enriched'].mean() > 0 else float('inf')
        else:
            orig_ir = np.nan

        r5_results.append({
            'gene': gene,
            'modality': mod_name,
            'n_variants': len(gdf),
            'IR_original': orig_ir,
            'IR_mean': result['mean'],
            'IR_median': result['median'],
            'CI_lower': result['ci_lower'],
            'CI_upper': result['ci_upper'],
            'includes_1': result['includes_1'],
            'n_valid_boots': result['n_valid']
        })

r5_boot = pd.DataFrame(r5_results)
r5_boot.to_csv('{}/prompt6_gene_IR_bootstrap_R5.tsv'.format(OUT), sep='\t', index=False)
print("R5 bootstrap complete: {} gene×modality results".format(len(r5_boot)))


print("\n\n--- 6-3. R4 vs R5 Gene IR Comparison ---")

comparison = []
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    r4_mod = r4_boot[r4_boot['modality'] == mod_name].set_index('gene')
    r5_mod = r5_boot[r5_boot['modality'] == mod_name].set_index('gene')

    common = r4_mod.index.intersection(r5_mod.index)
    print("\n  {} ({} common genes):".format(mod_name, len(common)))

    n_r4_in_r5_ci = 0
    n_r5_in_r4_ci = 0
    n_same_dir = 0
    n_both_sig = 0

    for gene in common:
        r4 = r4_mod.loc[gene]
        r5 = r5_mod.loc[gene]

        r4_in_r5 = r5['CI_lower'] <= r4['IR_median'] <= r5['CI_upper']
        r5_in_r4 = r4['CI_lower'] <= r5['IR_median'] <= r4['CI_upper']
        same_dir = (r4['IR_median'] > 1 and r5['IR_median'] > 1) or \
                   (r4['IR_median'] < 1 and r5['IR_median'] < 1)
        both_sig = not r4['includes_1'] and not r5['includes_1']

        if r4_in_r5:
            n_r4_in_r5_ci += 1
        if r5_in_r4:
            n_r5_in_r4_ci += 1
        if same_dir:
            n_same_dir += 1
        if both_sig:
            n_both_sig += 1

        comparison.append({
            'gene': gene, 'modality': mod_name,
            'R4_IR': r4['IR_median'], 'R4_CI_lower': r4['CI_lower'],
            'R4_CI_upper': r4['CI_upper'], 'R4_sig': not r4['includes_1'],
            'R5_IR': r5['IR_median'], 'R5_CI_lower': r5['CI_lower'],
            'R5_CI_upper': r5['CI_upper'], 'R5_sig': not r5['includes_1'],
            'R4_in_R5_CI': r4_in_r5, 'R5_in_R4_CI': r5_in_r4,
            'Same_direction': same_dir, 'Both_significant': both_sig
        })

    if len(common) > 0:
        print("    R4 IR in R5 CI: {}/{} ({:.1f}%)".format(
            n_r4_in_r5_ci, len(common), n_r4_in_r5_ci / len(common) * 100))
        print("    R5 IR in R4 CI: {}/{} ({:.1f}%)".format(
            n_r5_in_r4_ci, len(common), n_r5_in_r4_ci / len(common) * 100))
        print("    Same direction: {}/{} ({:.1f}%)".format(
            n_same_dir, len(common), n_same_dir / len(common) * 100))
        print("    Both CI exclude 1: {}".format(n_both_sig))

        r4_irs = [r4_mod.loc[g, 'IR_median'] for g in common]
        r5_irs = [r5_mod.loc[g, 'IR_median'] for g in common]
        r4_irs_clean = [x for x, y in zip(r4_irs, r5_irs) if np.isfinite(x) and np.isfinite(y)]
        r5_irs_clean = [y for x, y in zip(r4_irs, r5_irs) if np.isfinite(x) and np.isfinite(y)]
        if len(r4_irs_clean) >= 5:
            rho, pval = stats.spearmanr(r4_irs_clean, r5_irs_clean)
            print("    Spearman r: {:.3f} (P={:.4f})".format(rho, pval))

comp_df = pd.DataFrame(comparison)
comp_df.to_csv('{}/prompt6_R4_vs_R5_gene_IR.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- 6-4. Empirical Bayes Shrinkage ---")

shrinkage_results = []
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    mod_data = r4_boot[r4_boot['modality'] == mod_name].copy()

    if len(mod_data) == 0:
        continue

    grand_mean = np.average(mod_data['IR_median'],
                            weights=mod_data['n_variants'])

    grand_var = mod_data['IR_median'].var()

    print("\n  {} (grand mean={:.3f}, grand var={:.4f}):".format(
        mod_name, grand_mean, grand_var))

    for _, row in mod_data.iterrows():
        ci_width = row['CI_upper'] - row['CI_lower']
        within_var = (ci_width / 3.92) ** 2

        if grand_var + within_var > 0:
            shrink_factor = grand_var / (grand_var + within_var)
        else:
            shrink_factor = 0.5

        shrunken_ir = grand_mean + shrink_factor * (row['IR_median'] - grand_mean)

        shrinkage_results.append({
            'gene': row['gene'],
            'modality': mod_name,
            'n_variants': row['n_variants'],
            'IR_raw': row['IR_median'],
            'IR_shrunken': shrunken_ir,
            'shrink_factor': shrink_factor,
            'CI_lower': row['CI_lower'],
            'CI_upper': row['CI_upper']
        })

shrink_df = pd.DataFrame(shrinkage_results)
shrink_df.to_csv('{}/prompt6_shrinkage_IR.tsv'.format(OUT), sep='\t', index=False)

rna_shrink = shrink_df[shrink_df['modality'] == 'rna_seq']
print("\n  RNA-seq shrinkage summary:")
print("    Raw IR range: {:.3f} - {:.3f}".format(
    rna_shrink['IR_raw'].min(), rna_shrink['IR_raw'].max()))
print("    Shrunken IR range: {:.3f} - {:.3f}".format(
    rna_shrink['IR_shrunken'].min(), rna_shrink['IR_shrunken'].max()))
print("    Mean shrinkage factor: {:.3f}".format(rna_shrink['shrink_factor'].mean()))


print("\n\n--- 6-5. Special Genes Highlight ---")

print("\nReversed pattern genes (expected IR < 1):")
for gene in ['APH1B', 'CASP7', 'CD2AP']:
    rna = r4_boot[(r4_boot['gene'] == gene) & (r4_boot['modality'] == 'rna_seq')]
    if len(rna) > 0:
        row = rna.iloc[0]
        sig = 'CI excludes 1' if not row['includes_1'] else 'CI includes 1'
        print("  {:8s} IR={:.3f} [{:.3f}-{:.3f}] N={:.0f}  {}".format(
            gene, row['IR_median'], row['CI_lower'], row['CI_upper'],
            row['n_variants'], sig))
    else:
        print("  {:8s} Not enough variants for bootstrap".format(gene))

print("\nStrong positive genes (expected IR > 1.5):")
for gene in ['SIGLEC11', 'TREML2', 'LILRB2', 'ICA1']:
    rna = r4_boot[(r4_boot['gene'] == gene) & (r4_boot['modality'] == 'rna_seq')]
    if len(rna) > 0:
        row = rna.iloc[0]
        sig = 'CI excludes 1' if not row['includes_1'] else 'CI includes 1'
        print("  {:12s} IR={:.3f} [{:.3f}-{:.3f}] N={:.0f}  {}".format(
            gene, row['IR_median'], row['CI_lower'], row['CI_upper'],
            row['n_variants'], sig))
    else:
        print("  {:12s} Not enough variants for bootstrap".format(gene))


print("\n\n--- Generating Forest Plot ---")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rna_r4_sorted = rna_r4.sort_values('IR_median')

    fig, ax = plt.subplots(figsize=(10, max(8, len(rna_r4_sorted) * 0.35)))

    y_pos = range(len(rna_r4_sorted))
    for i, (_, row) in enumerate(rna_r4_sorted.iterrows()):
        color = 'red' if not row['includes_1'] and row['IR_median'] > 1 else \
                'blue' if not row['includes_1'] and row['IR_median'] < 1 else 'gray'
        ax.errorbar(row['IR_median'], i,
                   xerr=[[row['IR_median'] - row['CI_lower']],
                         [row['CI_upper'] - row['IR_median']]],
                   fmt='o', color=color, markersize=4, capsize=2, linewidth=1)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rna_r4_sorted['gene'].values, fontsize=7)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Interaction Ratio (RNA-seq)')
    ax.set_title('R4 Gene-specific IR with 95% Bootstrap CI\n'
                 'Red=significant IR>1, Blue=significant IR<1, Gray=CI includes 1')
    plt.tight_layout()
    plt.savefig('{}/prompt6_forest_plot_R4.png'.format(OUT), dpi=150)
    plt.close()
    print("  Saved: prompt6_forest_plot_R4.png")

    comp_rna = comp_df[comp_df['modality'] == 'rna_seq'].copy()
    if len(comp_rna) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))

        for _, row in comp_rna.iterrows():
            color = 'green' if row['Same_direction'] else 'red'
            ax.errorbar(row['R4_IR'], row['R5_IR'],
                       xerr=[[row['R4_IR'] - row['R4_CI_lower']],
                             [row['R4_CI_upper'] - row['R4_IR']]],
                       yerr=[[row['R5_IR'] - row['R5_CI_lower']],
                             [row['R5_CI_upper'] - row['R5_IR']]],
                       fmt='o', color=color, markersize=4, capsize=2,
                       linewidth=0.5, alpha=0.7)

        lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim_max], [0, lim_max], 'k--', linewidth=0.8, alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.5)
        ax.axvline(x=1, color='gray', linestyle=':', linewidth=0.5)
        ax.set_xlabel('R4 IR (RNA-seq)')
        ax.set_ylabel('R5 IR (RNA-seq)')
        ax.set_title('R4 vs R5 Gene-specific IR\n'
                     'Green=same direction, Red=opposite')
        plt.tight_layout()
        plt.savefig('{}/prompt6_R4_R5_scatter.png'.format(OUT), dpi=150)
        plt.close()
        print("  Saved: prompt6_R4_R5_scatter.png")

except ImportError:
    print("  matplotlib not available, skipping plots")
except Exception as e:
    print("  Plot error: {}".format(e))


print("\n" + "=" * 70)
print("PROMPT 6 SUMMARY")
print("=" * 70)

rna_r4 = r4_boot[r4_boot['modality'] == 'rna_seq']
rna_r5 = r5_boot[r5_boot['modality'] == 'rna_seq']

print("\nR4 ({} genes, RNA-seq):".format(len(rna_r4)))
print("  CI excludes 1: {} genes".format((~rna_r4['includes_1']).sum()))
print("  IR > 1: {} genes, IR < 1: {} genes".format(
    (rna_r4['IR_median'] > 1).sum(), (rna_r4['IR_median'] < 1).sum()))

print("\nR5 ({} genes, RNA-seq):".format(len(rna_r5)))
print("  CI excludes 1: {} genes".format((~rna_r5['includes_1']).sum()))

comp_rna = comp_df[comp_df['modality'] == 'rna_seq']
if len(comp_rna) > 0:
    print("\nR4 vs R5 comparison ({} common genes, RNA-seq):".format(len(comp_rna)))
    print("  R4 IR in R5 CI: {}/{} ({:.1f}%)".format(
        comp_rna['R4_in_R5_CI'].sum(), len(comp_rna),
        comp_rna['R4_in_R5_CI'].mean() * 100))
    print("  Same direction: {}/{} ({:.1f}%)".format(
        comp_rna['Same_direction'].sum(), len(comp_rna),
        comp_rna['Same_direction'].mean() * 100))

print("\nFinished:", datetime.now())
