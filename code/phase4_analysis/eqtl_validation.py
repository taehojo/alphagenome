import pandas as pd
import numpy as np
import gzip
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = "<WORK_DIR>"
EQTLGEN_PATH = "<KBASE_GENOME_DIR>/GWAS/1/2019-12-11-cis-eQTLsFDR-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt.gz"
VARIANT_DATA = f"{PROJECT_DIR}/data/variant_cc_with_alphgenome.csv"
GENE_LIST = f"{PROJECT_DIR}/data/Supplementary_Table_S1_GeneList.csv"
OUTPUT_DIR = f"{PROJECT_DIR}/results/reviewer_response/analysis1_eqtl"

GTEX_BRAIN_PATHS = [
    f"{PROJECT_DIR}/data/external/Brain_Cortex.v8.signif_variant_gene_pairs.txt.gz",
    f"{PROJECT_DIR}/data/external/Brain_Hippocampus.v8.signif_variant_gene_pairs.txt.gz",
    f"{PROJECT_DIR}/data/external/Brain_Frontal_Cortex_BA9.v8.signif_variant_gene_pairs.txt.gz"
]


def load_variant_data():
    print("Loading variant data...")
    df = pd.read_csv(VARIANT_DATA)

    df['total_AC'] = df['case_AC'] + df['ctrl_AC']
    df_filtered = df[df['total_AC'] >= 3].copy()

    df_unique = df_filtered.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')

    print(f"  Total variants: {len(df)}")
    print(f"  After AC>=3 and dedup: {len(df_unique)}")

    df_unique['rsid'] = df_unique['variant_id'].apply(
        lambda x: x if x.startswith('rs') else None
    )
    n_rsid = df_unique['rsid'].notna().sum()
    print(f"  Variants with rsID: {n_rsid}")

    df_unique['pos_key'] = df_unique['chr_num'].astype(str) + ':' + df_unique['pos'].astype(str)

    return df_unique


def load_gene_list():
    gene_df = pd.read_csv(GENE_LIST)
    return set(gene_df['gene_name'].tolist())


def parse_eqtlgen_for_genes(eqtl_path, target_genes, chunk_size=500000):
    print(f"\nParsing eQTLGen data (blood eQTLs)...")
    print(f"  File: {eqtl_path}")
    print(f"  Target genes: {len(target_genes)}")

    eqtls = []
    total_rows = 0
    matched_rows = 0

    col_names = ['Pvalue', 'SNP', 'SNPChr', 'SNPPos', 'AssessedAllele', 'OtherAllele',
                 'Zscore', 'Gene', 'GeneSymbol', 'GeneChr', 'GenePos',
                 'NrCohorts', 'NrSamples', 'FDR', 'BonferroniP']

    with gzip.open(eqtl_path, 'rt') as f:
        header = f.readline()

        batch = []
        for line in f:
            total_rows += 1

            fields = line.strip().split('\t')
            if len(fields) < 15:
                continue

            gene_symbol = fields[8]

            if gene_symbol in target_genes:
                try:
                    eqtl_record = {
                        'rsid': fields[1],
                        'chr': fields[2],
                        'pos': int(fields[3]),
                        'assessed_allele': fields[4],
                        'other_allele': fields[5],
                        'zscore': float(fields[6]),
                        'gene_id': fields[7],
                        'gene_symbol': gene_symbol,
                        'pvalue': float(fields[0]) if fields[0] != 'NA' else None,
                        'fdr': float(fields[13]) if fields[13] != 'NA' else None,
                        'bonferroni_p': float(fields[14]) if fields[14] != 'NA' else None,
                        'n_samples': int(fields[12])
                    }
                    batch.append(eqtl_record)
                    matched_rows += 1
                except (ValueError, IndexError) as e:
                    continue

            if total_rows % 5000000 == 0:
                print(f"  Processed {total_rows/1e6:.1f}M rows, matched {matched_rows:,} eQTLs...")

            if len(batch) >= chunk_size:
                eqtls.extend(batch)
                batch = []

        eqtls.extend(batch)

    print(f"  Total rows processed: {total_rows:,}")
    print(f"  eQTLs for target genes: {len(eqtls):,}")

    return pd.DataFrame(eqtls)


def match_variants_with_eqtls(variant_df, eqtl_df):
    print("\nMatching variants with eQTLs...")

    eqtl_df['pos_key'] = eqtl_df['chr'].astype(str) + ':' + eqtl_df['pos'].astype(str)

    eqtl_rsids = set(eqtl_df['rsid'].dropna().unique())
    eqtl_positions = set(eqtl_df['pos_key'].dropna().unique())

    print(f"  Unique eQTL rsIDs: {len(eqtl_rsids):,}")
    print(f"  Unique eQTL positions: {len(eqtl_positions):,}")

    variant_df['is_eqtl_rsid'] = variant_df['rsid'].isin(eqtl_rsids)

    variant_df['is_eqtl_pos'] = variant_df['pos_key'].isin(eqtl_positions)

    variant_df['is_eqtl'] = variant_df['is_eqtl_rsid'] | variant_df['is_eqtl_pos']

    n_matched_rsid = variant_df['is_eqtl_rsid'].sum()
    n_matched_pos = variant_df['is_eqtl_pos'].sum()
    n_matched_any = variant_df['is_eqtl'].sum()

    print(f"\nMatching results:")
    print(f"  By rsID: {n_matched_rsid:,} ({n_matched_rsid/len(variant_df)*100:.1f}%)")
    print(f"  By position: {n_matched_pos:,} ({n_matched_pos/len(variant_df)*100:.1f}%)")
    print(f"  Any match: {n_matched_any:,} ({n_matched_any/len(variant_df)*100:.1f}%)")

    eqtl_effects = eqtl_df.groupby('rsid').agg({
        'zscore': lambda x: x.iloc[np.argmax(np.abs(x))],
        'pvalue': 'min',
        'gene_symbol': 'first'
    }).reset_index()
    eqtl_effects.columns = ['rsid', 'eqtl_zscore', 'eqtl_pvalue', 'eqtl_gene']

    variant_df = variant_df.merge(eqtl_effects, on='rsid', how='left')

    return variant_df


def calculate_eqtl_enrichment(variant_df):
    print("\n" + "="*70)
    print("eQTL Enrichment Analysis")
    print("="*70)

    case_enriched = variant_df[variant_df['enrichment'] == 'case_enriched']
    ctrl_enriched = variant_df[variant_df['enrichment'] == 'ctrl_enriched']

    case_eqtl_rate = case_enriched['is_eqtl'].mean()
    ctrl_eqtl_rate = ctrl_enriched['is_eqtl'].mean()

    print(f"\neQTL rates:")
    print(f"  Case-enriched: {case_eqtl_rate*100:.2f}% ({case_enriched['is_eqtl'].sum()}/{len(case_enriched)})")
    print(f"  Control-enriched: {ctrl_eqtl_rate*100:.2f}% ({ctrl_enriched['is_eqtl'].sum()}/{len(ctrl_enriched)})")


    a = case_enriched['is_eqtl'].sum()
    b = len(case_enriched) - a
    c = ctrl_enriched['is_eqtl'].sum()
    d = len(ctrl_enriched) - c

    contingency = [[a, b], [c, d]]
    print(f"\nContingency table:")
    print(f"                eQTL    non-eQTL")
    print(f"  Case-enriched  {a:<7} {b}")
    print(f"  Ctrl-enriched  {c:<7} {d}")

    odds_ratio, fisher_p = stats.fisher_exact(contingency)

    print(f"\nFisher's exact test:")
    print(f"  Odds ratio: {odds_ratio:.4f}")
    print(f"  P-value: {fisher_p:.4e}")

    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test:")
    print(f"  Chi2: {chi2:.4f}")
    print(f"  P-value: {chi_p:.4e}")

    if fisher_p < 0.05:
        if odds_ratio > 1:
            print(f"\n** Case-enriched variants are significantly MORE likely to be eQTLs (OR={odds_ratio:.2f}, P={fisher_p:.2e})")
        else:
            print(f"\n** Case-enriched variants are significantly LESS likely to be eQTLs (OR={odds_ratio:.2f}, P={fisher_p:.2e})")
    else:
        print(f"\n** No significant difference in eQTL rate between case and control-enriched variants (P={fisher_p:.2e})")

    results = {
        'case_eqtl_rate': case_eqtl_rate,
        'ctrl_eqtl_rate': ctrl_eqtl_rate,
        'case_n_eqtl': a,
        'case_n_total': len(case_enriched),
        'ctrl_n_eqtl': c,
        'ctrl_n_total': len(ctrl_enriched),
        'odds_ratio': odds_ratio,
        'fisher_p': fisher_p,
        'chi2': chi2,
        'chi_p': chi_p
    }

    return results


def alphgenome_eqtl_correlation(variant_df):
    print("\n" + "="*70)
    print("AlphaGenome vs eQTL Effect Size Correlation")
    print("="*70)

    df_matched = variant_df[(variant_df['is_eqtl']) & (variant_df['eqtl_zscore'].notna())].copy()
    print(f"Variants with both AlphaGenome and eQTL data: {len(df_matched)}")

    if len(df_matched) < 10:
        print("Insufficient data for correlation analysis")
        return {}

    modalities = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
    results = {}

    for mod in modalities:
        valid_data = df_matched[[mod, 'eqtl_zscore']].dropna()
        if len(valid_data) > 10:
            r, p = stats.spearmanr(valid_data[mod], np.abs(valid_data['eqtl_zscore']))
            results[mod] = {'spearman_r': r, 'p_value': p, 'n': len(valid_data)}
            print(f"  {mod}: r={r:.4f}, P={p:.4e}, n={len(valid_data)}")

    return results


def high_alphgenome_eqtl_enrichment(variant_df, top_percentile=20):
    print("\n" + "="*70)
    print(f"High AlphaGenome Score (top {top_percentile}%) eQTL Enrichment")
    print("="*70)

    results = {}
    modalities = ['rna_seq_effect', 'cage_effect', 'chip_histone_effect']

    for mod in modalities:
        valid_data = variant_df[variant_df[mod].notna()].copy()
        if len(valid_data) == 0:
            continue

        threshold = valid_data[mod].quantile(1 - top_percentile/100)

        high_score = valid_data[valid_data[mod] >= threshold]
        low_score = valid_data[valid_data[mod] < threshold]

        high_eqtl_rate = high_score['is_eqtl'].mean()
        low_eqtl_rate = low_score['is_eqtl'].mean()

        a = high_score['is_eqtl'].sum()
        b = len(high_score) - a
        c = low_score['is_eqtl'].sum()
        d = len(low_score) - c

        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])

        results[mod] = {
            'threshold': threshold,
            'high_n': len(high_score),
            'high_eqtl_rate': high_eqtl_rate,
            'low_n': len(low_score),
            'low_eqtl_rate': low_eqtl_rate,
            'odds_ratio': odds_ratio,
            'fisher_p': fisher_p
        }

        print(f"\n{mod}:")
        print(f"  Threshold (top {top_percentile}%): {threshold:.4f}")
        print(f"  High score eQTL rate: {high_eqtl_rate*100:.2f}% ({a}/{len(high_score)})")
        print(f"  Low score eQTL rate: {low_eqtl_rate*100:.2f}% ({c}/{len(low_score)})")
        print(f"  Odds ratio: {odds_ratio:.4f}, P={fisher_p:.4e}")

    return results


def generate_summary(enrichment_results, correlation_results, high_score_results, eqtl_df, variant_df):

    print("\n" + "="*70)
    print("Summary for Manuscript")
    print("="*70)

    print("\n** Key Findings **")

    total_variants = len(variant_df)
    n_eqtl = variant_df['is_eqtl'].sum()
    print(f"\n1. eQTL overlap:")
    print(f"   - {n_eqtl:,} of {total_variants:,} variants ({n_eqtl/total_variants*100:.1f}%) overlap with blood eQTLs")
    print(f"   - eQTL database contains {len(eqtl_df):,} eQTLs for 86 AD genes")

    print(f"\n2. Case-enriched vs Control-enriched eQTL rates:")
    print(f"   - Case-enriched: {enrichment_results['case_eqtl_rate']*100:.2f}%")
    print(f"   - Control-enriched: {enrichment_results['ctrl_eqtl_rate']*100:.2f}%")
    print(f"   - Odds ratio: {enrichment_results['odds_ratio']:.3f}")
    print(f"   - Fisher P: {enrichment_results['fisher_p']:.4e}")

    if correlation_results:
        print(f"\n3. AlphaGenome-eQTL effect size correlation:")
        for mod, res in correlation_results.items():
            print(f"   - {mod}: r={res['spearman_r']:.3f} (P={res['p_value']:.4e})")

    if high_score_results:
        print(f"\n4. High AlphaGenome score eQTL enrichment:")
        for mod, res in high_score_results.items():
            if res['fisher_p'] < 0.05:
                print(f"   - {mod}: OR={res['odds_ratio']:.2f}, P={res['fisher_p']:.4e} (SIGNIFICANT)")
            else:
                print(f"   - {mod}: OR={res['odds_ratio']:.2f}, P={res['fisher_p']:.4e}")

    print(f"\n** LIMITATION **")
    print(f"   eQTLGen represents peripheral blood eQTLs.")
    print(f"   Brain-specific eQTL validation (GTEx Brain) is warranted.")
    print(f"   Tissue-specific regulatory effects may differ.")


def main():
    print("="*70)
    print("Analysis 1: eQTL Cross-Validation (Blood eQTLs - eQTLGen)")
    print("="*70)

    variant_df = load_variant_data()
    target_genes = load_gene_list()

    eqtl_df = parse_eqtlgen_for_genes(EQTLGEN_PATH, target_genes)

    if len(eqtl_df) == 0:
        print("ERROR: No eQTLs found for target genes")
        return

    variant_df = match_variants_with_eqtls(variant_df, eqtl_df)

    enrichment_results = calculate_eqtl_enrichment(variant_df)

    correlation_results = alphgenome_eqtl_correlation(variant_df)

    high_score_results = high_alphgenome_eqtl_enrichment(variant_df)

    generate_summary(enrichment_results, correlation_results, high_score_results, eqtl_df, variant_df)

    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    output_cols = ['variant_id', 'gene_name', 'chr_num', 'pos', 'enrichment',
                   'cc_ratio', 'OR', 'is_eqtl', 'is_eqtl_rsid', 'is_eqtl_pos',
                   'eqtl_zscore', 'eqtl_pvalue', 'eqtl_gene',
                   'rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
    output_cols = [c for c in output_cols if c in variant_df.columns]
    variant_df[output_cols].to_csv(f"{OUTPUT_DIR}/variants_with_eqtl_annotation.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/variants_with_eqtl_annotation.csv")

    pd.DataFrame([enrichment_results]).to_csv(f"{OUTPUT_DIR}/eqtl_enrichment_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/eqtl_enrichment_results.csv")

    if correlation_results:
        corr_df = pd.DataFrame.from_dict(correlation_results, orient='index')
        corr_df.to_csv(f"{OUTPUT_DIR}/alphgenome_eqtl_correlation.csv")
        print(f"  Saved: {OUTPUT_DIR}/alphgenome_eqtl_correlation.csv")

    if high_score_results:
        high_df = pd.DataFrame.from_dict(high_score_results, orient='index')
        high_df.to_csv(f"{OUTPUT_DIR}/high_score_eqtl_enrichment.csv")
        print(f"  Saved: {OUTPUT_DIR}/high_score_eqtl_enrichment.csv")

    gene_eqtl_summary = eqtl_df.groupby('gene_symbol').agg({
        'rsid': 'count',
        'zscore': lambda x: np.mean(np.abs(x)),
        'pvalue': 'min'
    }).reset_index()
    gene_eqtl_summary.columns = ['gene', 'n_eqtls', 'mean_abs_zscore', 'min_pvalue']
    gene_eqtl_summary.to_csv(f"{OUTPUT_DIR}/gene_eqtl_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/gene_eqtl_summary.csv")

    print("\n" + "="*70)
    print("Analysis 1 Complete")
    print("="*70)


if __name__ == "__main__":
    main()
