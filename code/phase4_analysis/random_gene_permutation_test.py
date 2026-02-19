import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FULL_DATA = '<ADSP_AI_DIR>/LD-rarevariant-5th-all/analysis/comprehensive_analysis_FULL_1.8M/merged_plink_alphgenome_1.8M.csv'
AD_GENE_LIST = '<WORK_DIR>/data/Supplementary_Table_S1_GeneList.csv'
OUT_DIR = '<WORK_DIR>/results/continuous_analysis'

N_PERMUTATIONS = 1000
MIN_VARIANTS_PER_GENE = 10
AC_THRESHOLD = 3

print("=" * 60)
print("Random Gene Permutation Test")
print("=" * 60)

print("\n1. Loading AD gene list...")
ad_genes_df = pd.read_csv(AD_GENE_LIST)
ad_genes = set(ad_genes_df['gene_name'].str.upper().tolist())
print(f"   AD genes: {len(ad_genes)}")

print("\n2. Loading full variant data (1.8M variants)...")
print("   This may take a moment...")

chunks = []
for chunk in pd.read_csv(FULL_DATA, chunksize=100000):
    chunks.append(chunk)
df_full = pd.concat(chunks, ignore_index=True)
print(f"   Total variants loaded: {len(df_full):,}")

print("\n3. Calculating case-control ratios...")
df_full['cc_ratio'] = df_full['F_A'] / df_full['F_U']

df_full = df_full.replace([np.inf, -np.inf], np.nan)
df_full = df_full.dropna(subset=['cc_ratio', 'gene_name'])

N_CASES = 6296
N_CONTROLS = 18299
df_full['case_AC'] = (df_full['F_A'] * 2 * N_CASES).round()
df_full['ctrl_AC'] = (df_full['F_U'] * 2 * N_CONTROLS).round()
df_full['total_AC'] = df_full['case_AC'] + df_full['ctrl_AC']

df_full = df_full[df_full['total_AC'] >= AC_THRESHOLD].copy()
print(f"   Variants after AC>={AC_THRESHOLD} filter: {len(df_full):,}")

df_full['gene_name_upper'] = df_full['gene_name'].str.upper()

gene_counts = df_full.groupby('gene_name_upper').size()
valid_genes = gene_counts[gene_counts >= MIN_VARIANTS_PER_GENE].index.tolist()
print(f"   Genes with >={MIN_VARIANTS_PER_GENE} variants: {len(valid_genes)}")

ad_genes_in_data = [g for g in valid_genes if g in ad_genes]
non_ad_genes = [g for g in valid_genes if g not in ad_genes]
print(f"   AD genes in data: {len(ad_genes_in_data)}")
print(f"   Non-AD genes available for sampling: {len(non_ad_genes)}")

def calculate_case_enrichment(df, gene_list):
    gene_list_upper = [g.upper() for g in gene_list]
    subset = df[df['gene_name_upper'].isin(gene_list_upper)]

    if len(subset) == 0:
        return np.nan, 0

    pct_case_enriched = (subset['cc_ratio'] > 1).mean() * 100
    n_variants = len(subset)

    return pct_case_enriched, n_variants

print("\n4. Calculating observed case-enrichment for AD genes...")
observed_pct, observed_n = calculate_case_enrichment(df_full, ad_genes_in_data)
print(f"   AD genes: {observed_pct:.2f}% case-enriched ({observed_n:,} variants)")

print(f"\n5. Running permutation test ({N_PERMUTATIONS:,} iterations)...")
np.random.seed(42)

permutation_results = []
n_ad_genes = len(ad_genes_in_data)

for i in range(N_PERMUTATIONS):
    if (i + 1) % 100 == 0:
        print(f"   Iteration {i+1}/{N_PERMUTATIONS}")

    random_genes = np.random.choice(non_ad_genes, size=n_ad_genes, replace=False)

    pct, n = calculate_case_enrichment(df_full, random_genes)
    permutation_results.append({
        'iteration': i + 1,
        'pct_case_enriched': pct,
        'n_variants': n
    })

perm_df = pd.DataFrame(permutation_results)

mean_random = perm_df['pct_case_enriched'].mean()
std_random = perm_df['pct_case_enriched'].std()
min_random = perm_df['pct_case_enriched'].min()
max_random = perm_df['pct_case_enriched'].max()

p_value = (perm_df['pct_case_enriched'] >= observed_pct).mean()
if p_value == 0:
    p_value_str = f"< {1/N_PERMUTATIONS:.4f}"
else:
    p_value_str = f"= {p_value:.4f}"

z_score = (observed_pct - mean_random) / std_random

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nAD Genes (n={len(ad_genes_in_data)}):")
print(f"  Case-enriched: {observed_pct:.2f}%")
print(f"  N variants: {observed_n:,}")

print(f"\nRandom Gene Sets (n={N_PERMUTATIONS} permutations):")
print(f"  Mean: {mean_random:.2f}%")
print(f"  Std: {std_random:.2f}%")
print(f"  Range: {min_random:.2f}% - {max_random:.2f}%")

print(f"\nStatistical Test:")
print(f"  Z-score: {z_score:.2f}")
print(f"  P-value {p_value_str}")

results_summary = {
    'ad_genes_n': len(ad_genes_in_data),
    'ad_genes_pct_case_enriched': observed_pct,
    'ad_genes_n_variants': observed_n,
    'random_mean': mean_random,
    'random_std': std_random,
    'random_min': min_random,
    'random_max': max_random,
    'z_score': z_score,
    'p_value': p_value if p_value > 0 else 1/N_PERMUTATIONS,
    'n_permutations': N_PERMUTATIONS
}

pd.DataFrame([results_summary]).to_csv(f'{OUT_DIR}/permutation_test_summary.csv', index=False)
perm_df.to_csv(f'{OUT_DIR}/permutation_test_details.csv', index=False)

print(f"\nSaved: {OUT_DIR}/permutation_test_summary.csv")
print(f"Saved: {OUT_DIR}/permutation_test_details.csv")

print("\n6. Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(perm_df['pct_case_enriched'], bins=30, color='steelblue', alpha=0.7,
        edgecolor='white', label='Random gene sets')
ax.axvline(observed_pct, color='red', linewidth=2, linestyle='--',
           label=f'AD genes ({observed_pct:.1f}%)')
ax.axvline(mean_random, color='gray', linewidth=1.5, linestyle=':',
           label=f'Random mean ({mean_random:.1f}%)')

ax.set_xlabel('% Case-enriched Variants', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('A. Permutation Test: AD Genes vs Random Genes', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)

stats_text = f'Z = {z_score:.2f}\nP {p_value_str}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axes[1]

box_data = [perm_df['pct_case_enriched'].values, [observed_pct]]
positions = [1, 2]

bp = ax.boxplot([perm_df['pct_case_enriched'].values], positions=[1], widths=0.6,
                patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][0].set_alpha(0.7)

ax.scatter([2], [observed_pct], color='red', s=200, marker='*', zorder=5,
           label=f'AD genes ({observed_pct:.1f}%)')

ax.set_xticks([1, 2])
ax.set_xticklabels(['Random\nGene Sets\n(n=1000)', 'AD Genes\n(n=85)'], fontsize=11)
ax.set_ylabel('% Case-enriched Variants', fontsize=12)
ax.set_title('B. Comparison: Random vs AD Genes', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 2.5)

y_max = max(observed_pct, perm_df['pct_case_enriched'].max()) + 2
ax.plot([1, 1, 2, 2], [y_max-1, y_max, y_max, y_max-1], 'k-', linewidth=1)
ax.text(1.5, y_max + 0.5, f'P {p_value_str}', ha='center', fontsize=11)

ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/FigureS5_permutation_test.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/FigureS5_permutation_test.pdf', bbox_inches='tight')
plt.close()

print(f"Saved: {OUT_DIR}/FigureS5_permutation_test.png")
print(f"Saved: {OUT_DIR}/FigureS5_permutation_test.pdf")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if observed_pct > mean_random + 2 * std_random:
    print(f"\nAD genes show SIGNIFICANTLY HIGHER case-enrichment than random genes.")
    print(f"The observed {observed_pct:.1f}% is {z_score:.1f} standard deviations above")
    print(f"the random expectation of {mean_random:.1f}%.")
else:
    print(f"\nNo significant difference between AD genes and random genes.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
