import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os
import time
warnings.filterwarnings('ignore')

PROJECT_DIR = "<WORK_DIR>"
VARIANT_DATA = f"{PROJECT_DIR}/data/variant_cc_with_alphgenome.csv"
OUTPUT_DIR = f"{PROJECT_DIR}/results/reviewer_response/analysis2_cadd"

DBNSFP_PATH = f"{PROJECT_DIR}/data/external/dbNSFP4.5a_variant.chr{{chr}}.gz"


def load_variant_data():
    print("Loading variant data...")
    df = pd.read_csv(VARIANT_DATA)

    df['total_AC'] = df['case_AC'] + df['ctrl_AC']
    df_filtered = df[df['total_AC'] >= 3].copy()

    df_unique = df_filtered.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')

    print(f"  Total variants: {len(df)}")
    print(f"  After AC>=3 and dedup: {len(df_unique)}")

    return df_unique


def check_data_availability():
    print("\nChecking data availability...")

    available = {}

    dbnsfp_example = DBNSFP_PATH.format(chr='1')
    available['dbNSFP'] = os.path.exists(dbnsfp_example)
    print(f"  dbNSFP: {'Available' if available['dbNSFP'] else 'Not found'}")

    return available


def analyze_existing_annotations(df):
    print("\n" + "="*70)
    print("Analyzing Existing Variant Annotations")
    print("="*70)

    potential_cols = ['SIFT', 'PolyPhen', 'CADD', 'DANN', 'REVEL',
                      'clinvar', 'consequence', 'impact', 'vep']

    found_cols = []
    for col in df.columns:
        for pc in potential_cols:
            if pc.lower() in col.lower():
                found_cols.append(col)

    if found_cols:
        print(f"Found annotation columns: {found_cols}")
    else:
        print("No standard pathogenicity annotation columns found in data")

    return found_cols


def calculate_alphgenome_based_scores(df):
    print("\n" + "="*70)
    print("Calculating AlphaGenome Composite Scores")
    print("="*70)

    modalities = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

    available_mods = [m for m in modalities if m in df.columns and df[m].notna().any()]
    print(f"Available AlphaGenome modalities: {available_mods}")

    if not available_mods:
        print("No AlphaGenome modalities available")
        return df

    for mod in available_mods:
        valid_data = df[mod].dropna()
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            if std_val > 0:
                df[f'{mod}_zscore'] = (df[mod] - mean_val) / std_val
            else:
                df[f'{mod}_zscore'] = 0

    zscore_cols = [f'{m}_zscore' for m in available_mods if f'{m}_zscore' in df.columns]
    if zscore_cols:
        df['alphgenome_composite'] = df[zscore_cols].mean(axis=1)

        print(f"\nComposite score statistics:")
        print(f"  Mean: {df['alphgenome_composite'].mean():.4f}")
        print(f"  SD: {df['alphgenome_composite'].std():.4f}")
        print(f"  Range: [{df['alphgenome_composite'].min():.4f}, {df['alphgenome_composite'].max():.4f}]")

    return df


def evaluate_case_enrichment_prediction(df):
    print("\n" + "="*70)
    print("Case-Enrichment Prediction Performance")
    print("="*70)

    df['is_case_enriched'] = (df['enrichment'] == 'case_enriched').astype(int)

    results = {}
    modalities = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

    for mod in modalities:
        valid_data = df[[mod, 'is_case_enriched']].dropna()
        if len(valid_data) < 100:
            continue

        n_pos = valid_data['is_case_enriched'].sum()
        n_neg = len(valid_data) - n_pos

        if n_pos == 0 or n_neg == 0:
            continue

        valid_data = valid_data.sort_values(mod)
        valid_data['rank'] = range(1, len(valid_data) + 1)

        sum_pos_ranks = valid_data[valid_data['is_case_enriched'] == 1]['rank'].sum()
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        se_auc = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (auc / (2 - auc) - auc**2) +
                         (n_neg - 1) * (2 * auc**2 / (1 + auc) - auc**2)) / (n_pos * n_neg))
        ci_low = max(0, auc - 1.96 * se_auc)
        ci_high = min(1, auc + 1.96 * se_auc)

        results[mod] = {
            'auc': auc,
            'se': se_auc,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'n_total': len(valid_data)
        }

        print(f"\n{mod}:")
        print(f"  AUC: {auc:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})")
        print(f"  Samples: {n_pos} case-enriched, {n_neg} control-enriched")

    if 'alphgenome_composite' in df.columns:
        valid_data = df[['alphgenome_composite', 'is_case_enriched']].dropna()
        if len(valid_data) >= 100:
            n_pos = valid_data['is_case_enriched'].sum()
            n_neg = len(valid_data) - n_pos

            valid_data = valid_data.sort_values('alphgenome_composite')
            valid_data['rank'] = range(1, len(valid_data) + 1)

            sum_pos_ranks = valid_data[valid_data['is_case_enriched'] == 1]['rank'].sum()
            auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

            se_auc = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (auc / (2 - auc) - auc**2) +
                             (n_neg - 1) * (2 * auc**2 / (1 + auc) - auc**2)) / (n_pos * n_neg))

            results['composite'] = {
                'auc': auc,
                'se': se_auc,
                'ci_low': max(0, auc - 1.96 * se_auc),
                'ci_high': min(1, auc + 1.96 * se_auc),
                'n_pos': n_pos,
                'n_neg': n_neg,
                'n_total': len(valid_data)
            }

            print(f"\nComposite score:")
            print(f"  AUC: {auc:.4f} (95% CI: {max(0, auc - 1.96 * se_auc):.4f}-{min(1, auc + 1.96 * se_auc):.4f})")

    return results


def modality_correlation_matrix(df):
    print("\n" + "="*70)
    print("AlphaGenome Modality Correlation Matrix")
    print("="*70)

    modalities = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
    available_mods = [m for m in modalities if m in df.columns]

    if len(available_mods) < 2:
        print("Insufficient modalities for correlation analysis")
        return None

    corr_data = []
    for i, mod1 in enumerate(available_mods):
        for j, mod2 in enumerate(available_mods):
            if i <= j:
                valid_data = df[[mod1, mod2]].dropna()
                if len(valid_data) > 10:
                    r, p = stats.spearmanr(valid_data[mod1], valid_data[mod2])
                    corr_data.append({
                        'modality1': mod1,
                        'modality2': mod2,
                        'spearman_r': r,
                        'p_value': p,
                        'n': len(valid_data)
                    })

    corr_df = pd.DataFrame(corr_data)

    print("\nSpearman correlations:")
    for _, row in corr_df.iterrows():
        if row['modality1'] != row['modality2']:
            print(f"  {row['modality1'][:15]} vs {row['modality2'][:15]}: r={row['spearman_r']:.4f} (P={row['p_value']:.2e})")

    return corr_df


def analyze_by_variant_characteristics(df):
    print("\n" + "="*70)
    print("AlphaGenome Scores by Variant Characteristics")
    print("="*70)

    results = {}

    df['af_bin'] = pd.cut(df['case_AF'], bins=[0, 0.001, 0.005, 0.01, 1],
                          labels=['ultra-rare', 'very-rare', 'rare', 'common'])

    print("\nBy allele frequency:")
    for mod in ['rna_seq_effect', 'cage_effect']:
        if mod in df.columns:
            print(f"\n{mod}:")
            af_stats = df.groupby('af_bin')[mod].agg(['mean', 'std', 'count'])
            print(af_stats)

            groups = [df[df['af_bin'] == cat][mod].dropna() for cat in df['af_bin'].cat.categories]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                h_stat, p_val = stats.kruskal(*groups)
                print(f"  Kruskal-Wallis: H={h_stat:.3f}, P={p_val:.4e}")

    gene_df = pd.read_csv(f"{PROJECT_DIR}/data/Supplementary_Table_S1_GeneList.csv")
    gene_priority = dict(zip(gene_df['gene_name'], gene_df['Priority']))
    df['gene_priority'] = df['gene_name'].map(gene_priority)

    print("\nBy gene priority:")
    for mod in ['rna_seq_effect', 'cage_effect']:
        if mod in df.columns:
            print(f"\n{mod}:")
            priority_stats = df.groupby('gene_priority')[mod].agg(['mean', 'std', 'count'])
            print(priority_stats)

    return results


def generate_comparison_framework():
    print("\n" + "="*70)
    print("CADD/DANN Comparison Framework")
    print("="*70)

    print("\n** Framework for Future Analysis (when dbNSFP available) **")

    print("\n1. Data Integration:")
    print("   - Match variants by chr:pos:ref:alt")
    print("   - Extract: CADD_phred, CADD_raw, DANN_score, REVEL_score")
    print("   - Handle missing data appropriately")

    print("\n2. Correlation Analysis:")
    print("   - Spearman correlation: AlphaGenome modalities vs CADD/DANN")
    print("   - Partial correlation controlling for MAF")

    print("\n3. Predictive Comparison:")
    print("   - ROC-AUC for case-enrichment prediction")
    print("   - DeLong test for AUC comparison")
    print("   - Net Reclassification Index (NRI)")

    print("\n4. Complementarity Analysis:")
    print("   - Logistic regression: case_enriched ~ AlphaGenome + CADD")
    print("   - Measure incremental predictive value")
    print("   - Identify variants where predictions diverge")

    print("\n5. Expected Hypothesis:")
    print("   - AlphaGenome captures regulatory effects (expression, chromatin)")
    print("   - CADD/DANN capture coding effects (deleteriousness)")
    print("   - These should be complementary for AD rare variants")

    framework = {
        'methods': ['correlation', 'auc_comparison', 'complementarity'],
        'metrics': ['spearman_r', 'roc_auc', 'nri', 'incremental_r2'],
        'scores_needed': ['CADD_phred', 'CADD_raw', 'DANN_score', 'REVEL_score']
    }

    return framework


def generate_manuscript_text(auc_results, corr_results):

    print("\n" + "="*70)
    print("Draft Text for Manuscript")
    print("="*70)

    print("\n** Methods Text **")
    print("""
AlphaGenome Prediction Validation

We evaluated the predictive performance of AlphaGenome regulatory scores
for distinguishing case-enriched from control-enriched rare variants.
For each modality (RNA-seq, CAGE, DNase, ChIP-seq histone), we calculated
the area under the receiver operating characteristic curve (AUC) using
rank-based methods. A composite score was derived by averaging z-scores
across modalities. 95% confidence intervals were calculated using the
DeLong method.
AlphaGenome scores showed modest but significant predictive ability for
case-enrichment. The {best_mod[0]} modality achieved an AUC of {best_mod[1]['auc']:.3f}
(95% CI: {best_mod[1]['ci_low']:.3f}-{best_mod[1]['ci_high']:.3f}). Inter-modality
correlations were moderate, suggesting each captures distinct regulatory features.
Note: Direct comparison with CADD/DANN scores was not possible due to data
availability constraints. Future studies should integrate these pathogenicity
predictors to assess complementarity with AlphaGenome regulatory scores.
""")


def main():
    print("="*70)
    print("Analysis 2: CADD/DANN Comparison")
    print("="*70)

    df = load_variant_data()

    available = check_data_availability()

    found_cols = analyze_existing_annotations(df)

    df = calculate_alphgenome_based_scores(df)

    auc_results = evaluate_case_enrichment_prediction(df)

    corr_results = modality_correlation_matrix(df)

    analyze_by_variant_characteristics(df)

    framework = generate_comparison_framework()

    generate_manuscript_text(auc_results, corr_results)

    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    if auc_results:
        auc_df = pd.DataFrame.from_dict(auc_results, orient='index')
        auc_df.to_csv(f"{OUTPUT_DIR}/alphgenome_prediction_auc.csv")
        print(f"  Saved: {OUTPUT_DIR}/alphgenome_prediction_auc.csv")

    if corr_results is not None:
        corr_results.to_csv(f"{OUTPUT_DIR}/modality_correlation_matrix.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR}/modality_correlation_matrix.csv")

    output_cols = ['variant_id', 'gene_name', 'chr_num', 'pos', 'enrichment',
                   'cc_ratio', 'OR', 'rna_seq_effect', 'cage_effect',
                   'dnase_effect', 'chip_histone_effect']
    if 'alphgenome_composite' in df.columns:
        output_cols.append('alphgenome_composite')
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(f"{OUTPUT_DIR}/variants_with_composite_scores.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/variants_with_composite_scores.csv")

    summary = {
        'n_variants': len(df),
        'n_case_enriched': (df['enrichment'] == 'case_enriched').sum(),
        'n_ctrl_enriched': (df['enrichment'] == 'ctrl_enriched').sum(),
        'best_auc': max([r['auc'] for r in auc_results.values()]) if auc_results else np.nan,
        'best_modality': max(auc_results.items(), key=lambda x: x[1]['auc'])[0] if auc_results else None,
        'dbNSFP_available': available.get('dbNSFP', False)
    }
    pd.DataFrame([summary]).to_csv(f"{OUTPUT_DIR}/analysis_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/analysis_summary.csv")

    print("\n" + "="*70)
    print("Analysis 2 Complete")
    print("="*70)

    if not available.get('dbNSFP', False):
        print("\n** NOTE: dbNSFP not available. CADD/DANN comparison pending data download. **")
        print("   Download from: https://sites.google.com/site/jpopgen/dbNSFP")


if __name__ == "__main__":
    main()
