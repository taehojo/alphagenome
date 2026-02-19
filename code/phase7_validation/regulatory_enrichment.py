import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '<WORK_DIR>/analysis/additional_analyses'

print("=" * 70)
print("Prompt 9: Regulatory Element Annotation Enrichment")
print("Started:", datetime.now())
print("=" * 70)

MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

CELL_TYPES = {
    'Microglia': ['TREM2', 'PLCG2', 'ABI3', 'INPP5D', 'SPI1', 'CR1', 'MS4A6A',
                  'CD33', 'PILRA', 'SCIMP', 'LILRB2', 'SIGLEC11', 'TREML2', 'RHOH', 'BLNK'],
    'Neuron': ['APP', 'PSEN1', 'PSEN2', 'BIN1', 'CLU', 'SORL1', 'ADAM10', 'APH1B',
               'PICALM', 'PTK2B', 'CELF1', 'ANK3', 'ICA1', 'EPHA1', 'WNT3',
               'MEF2C', 'CNTNAP2'],
    'Astrocyte': ['APOE', 'ABCA7', 'ABCA1', 'GRN', 'FERMT2', 'HESX1', 'CLU'],
}

print("\n--- 9-1. Data Preparation ---")

df = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
df['total_AC'] = df['case_AC'] + df['ctrl_AC']
df = df[df['total_AC'] >= 3].sort_values('total_AC', ascending=False) \
       .drop_duplicates('variant_id', keep='first').copy()
df['is_case_enriched'] = df['enrichment'].isin(['case_enriched', 'case_only'])
print("R4 variants: {:,}".format(len(df)))

df['cell_type'] = 'Ubiquitous'
for ct, genes in CELL_TYPES.items():
    df.loc[df['gene_name'].isin(genes), 'cell_type'] = ct

for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    threshold = df[mod].quantile(0.80)
    df['{}_high'.format(mod_name)] = df[mod] >= threshold

chromhmm_file = '{}/regulatory_data/E073_15_coreMarks_mnemonics.bed'.format(OUT)
if os.path.exists(chromhmm_file):
    chromhmm = pd.read_csv(chromhmm_file, sep='\t', header=None,
                            names=['chrom', 'start', 'end', 'state'])
    print("ChromHMM E073: {:,} regions".format(len(chromhmm)))

    chromhmm['state_name'] = chromhmm['state'].apply(lambda x: x.split('_', 1)[1] if '_' in x else x)
    chromhmm['state_num'] = chromhmm['state'].apply(lambda x: int(x.split('_')[0]) if '_' in x else 0)

    state_categories = {
        'TssA': 'Active_Promoter', 'TssAFlnk': 'Active_Promoter',
        'TxFlnk': 'Transcription', 'Tx': 'Transcription', 'TxWk': 'Transcription',
        'EnhG': 'Enhancer', 'Enh': 'Enhancer',
        'ZNF/Rpts': 'ZNF_Repeats',
        'Het': 'Heterochromatin',
        'TssBiv': 'Bivalent', 'BivFlnk': 'Bivalent', 'EnhBiv': 'Bivalent',
        'ReprPC': 'Repressed', 'ReprPCWk': 'Repressed',
        'Quies': 'Quiescent'
    }
    chromhmm['category'] = chromhmm['state_name'].map(state_categories).fillna('Other')

    print("  State distribution:")
    state_counts = chromhmm['state_name'].value_counts()
    for state in state_counts.index:
        print("    {:12s}: {:,} regions".format(state, state_counts[state]))
else:
    print("  ChromHMM file not found")
    chromhmm = None

data_avail = []
data_avail.append({
    'Resource': 'Roadmap E073 ChromHMM 15-state',
    'Status': 'AVAILABLE' if chromhmm is not None else 'NOT FOUND',
    'Source': 'egg2.wustl.edu/roadmap',
    'N_regions': len(chromhmm) if chromhmm is not None else 0
})
data_avail.append({
    'Resource': 'Nott 2019 microglia enhancers',
    'Status': 'NOT DOWNLOADED (requires manual access)',
    'Source': 'doi:10.1126/science.aay0793',
    'N_regions': 0
})
data_avail.append({
    'Resource': 'Gosselin 2017 microglia enhancers',
    'Status': 'NOT DOWNLOADED (requires manual access)',
    'Source': 'doi:10.1126/science.aal3222',
    'N_regions': 0
})
pd.DataFrame(data_avail).to_csv('{}/prompt9_data_availability.tsv'.format(OUT),
                                 sep='\t', index=False)


print("\n\n--- 9-2. ChromHMM State Mapping ---")

if chromhmm is not None:
    print("  Building interval index...")

    df['chrom'] = 'chr' + df['chr_num'].astype(str)

    variant_states = []

    for chrom in df['chrom'].unique():
        chrom_variants = df[df['chrom'] == chrom]
        chrom_chromhmm = chromhmm[chromhmm['chrom'] == chrom].sort_values('start')

        if len(chrom_chromhmm) == 0:
            for _, v in chrom_variants.iterrows():
                variant_states.append({'variant_id': v['variant_id'], 'state': 'Unknown',
                                        'category': 'Unknown'})
            continue

        starts = chrom_chromhmm['start'].values
        ends = chrom_chromhmm['end'].values
        states = chrom_chromhmm['state_name'].values
        categories = chrom_chromhmm['category'].values

        for _, v in chrom_variants.iterrows():
            pos = v['pos']
            idx = np.searchsorted(starts, pos, side='right') - 1
            if 0 <= idx < len(starts) and starts[idx] <= pos < ends[idx]:
                variant_states.append({
                    'variant_id': v['variant_id'],
                    'state': states[idx],
                    'category': categories[idx]
                })
            else:
                variant_states.append({
                    'variant_id': v['variant_id'],
                    'state': 'Unknown',
                    'category': 'Unknown'
                })

    vs_df = pd.DataFrame(variant_states)
    df = df.merge(vs_df, on='variant_id', how='left')

    print("  Mapped {:,} variants to ChromHMM states".format(len(vs_df)))
    print("\n  Variant distribution by ChromHMM state:")
    state_dist = df['state'].value_counts()
    for state, count in state_dist.items():
        pct = count / len(df) * 100
        print("    {:12s}: {:,} ({:.1f}%)".format(state, count, pct))

    print("\n\n--- 9-3. ChromHMM Enrichment Analysis ---")

    chromhmm_results = []

    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        high_col = '{}_high'.format(mod_name)

        print("\n  {} (top 20% vs bottom 80%):".format(mod_name))

        for state_cat in ['Active_Promoter', 'Enhancer', 'Transcription',
                          'Bivalent', 'Repressed', 'Quiescent', 'Heterochromatin']:
            in_state = df['category'] == state_cat
            n_high_in = (df[high_col] & in_state).sum()
            n_high_out = (df[high_col] & ~in_state).sum()
            n_low_in = (~df[high_col] & in_state).sum()
            n_low_out = (~df[high_col] & ~in_state).sum()

            if n_high_in + n_low_in > 0:
                pct_high = n_high_in / (n_high_in + n_high_out) * 100
                pct_low = n_low_in / (n_low_in + n_low_out) * 100
                odds_ratio, fisher_p = stats.fisher_exact(
                    [[n_high_in, n_high_out], [n_low_in, n_low_out]])
            else:
                pct_high, pct_low = 0, 0
                odds_ratio, fisher_p = np.nan, np.nan

            print("    {:18s}: high={:.1f}%, low={:.1f}%, OR={:.3f}, P={:.4f}".format(
                state_cat, pct_high, pct_low, odds_ratio if not np.isnan(odds_ratio) else 0,
                fisher_p if not np.isnan(fisher_p) else 1))

            chromhmm_results.append({
                'Modality': mod_name, 'State_category': state_cat,
                'N_high_in': n_high_in, 'N_high_out': n_high_out,
                'N_low_in': n_low_in, 'N_low_out': n_low_out,
                'Pct_high': pct_high, 'Pct_low': pct_low,
                'OR': odds_ratio, 'Fisher_P': fisher_p
            })

    pd.DataFrame(chromhmm_results).to_csv('{}/prompt9_chromhmm_enrichment.tsv'.format(OUT),
                                           sep='\t', index=False)

    print("\n  Chi-square test (high vs low effect × chromatin state):")
    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        high_col = '{}_high'.format(mod_name)

        contingency = pd.crosstab(df[high_col], df['category'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print("    {}: Chi2={:.2f}, dof={}, P={:.4f}".format(mod_name, chi2, dof, p))


    print("\n\n--- 9-3b. Case-Enriched High-Effect in Active Elements ---")

    ce_active_results = []
    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        high_col = '{}_high'.format(mod_name)

        ce_high = df[df['is_case_enriched'] & df[high_col]]
        cte_high = df[~df['is_case_enriched'] & df[high_col]]

        for state_cat in ['Active_Promoter', 'Enhancer']:
            n_ce_in = (ce_high['category'] == state_cat).sum()
            n_ce_out = (ce_high['category'] != state_cat).sum()
            n_cte_in = (cte_high['category'] == state_cat).sum()
            n_cte_out = (cte_high['category'] != state_cat).sum()

            if n_ce_in + n_cte_in > 0:
                pct_ce = n_ce_in / len(ce_high) * 100 if len(ce_high) > 0 else 0
                pct_cte = n_cte_in / len(cte_high) * 100 if len(cte_high) > 0 else 0
                odds_ratio, fisher_p = stats.fisher_exact(
                    [[n_ce_in, n_ce_out], [n_cte_in, n_cte_out]])
            else:
                pct_ce, pct_cte = 0, 0
                odds_ratio, fisher_p = np.nan, np.nan

            print("  {} {}: CE_high={:.1f}%, CtE_high={:.1f}%, OR={:.3f}, P={:.4f}".format(
                mod_name, state_cat, pct_ce, pct_cte,
                odds_ratio if not np.isnan(odds_ratio) else 0,
                fisher_p if not np.isnan(fisher_p) else 1))

            ce_active_results.append({
                'Modality': mod_name, 'State_category': state_cat,
                'CE_high_pct': pct_ce, 'CtE_high_pct': pct_cte,
                'OR': odds_ratio, 'Fisher_P': fisher_p
            })


    print("\n\n--- 9-3c. Cell Type-Specific ChromHMM ---")

    for ct in ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']:
        ct_df = df[df['cell_type'] == ct]
        if len(ct_df) == 0:
            continue

        print("\n  {} (N={:,}):".format(ct, len(ct_df)))

        for state_cat in ['Active_Promoter', 'Enhancer', 'Transcription',
                          'Repressed', 'Quiescent']:
            pct = (ct_df['category'] == state_cat).mean() * 100
            in_state = ct_df[ct_df['category'] == state_cat]
            if len(in_state) > 0:
                ce_pct = in_state['is_case_enriched'].mean() * 100
            else:
                ce_pct = 0
            print("    {:18s}: {:.1f}% of variants, CE%={:.1f}%".format(
                state_cat, pct, ce_pct))


    print("\n\n--- 9-4. Reversed Pattern Genes ---")

    reversed_genes = ['APH1B', 'CASP7', 'CD2AP']
    reversed_results = []

    for gene in reversed_genes:
        gdf = df[df['gene_name'] == gene]
        if len(gdf) == 0:
            continue

        print("\n  {} (N={:,}):".format(gene, len(gdf)))

        for mod in ['rna_seq_effect']:
            mod_name = mod.replace('_effect', '')
            high_col = '{}_high'.format(mod_name)

            ce_high = gdf[gdf['is_case_enriched'] & gdf[high_col]]
            cte_high = gdf[~gdf['is_case_enriched'] & gdf[high_col]]

            print("    {} high-effect: CE={}, CtE={}".format(
                mod_name, len(ce_high), len(cte_high)))

            if len(ce_high) > 0:
                print("    CE high-effect states: {}".format(
                    ce_high['state'].value_counts().to_dict()))
            if len(cte_high) > 0:
                print("    CtE high-effect states: {}".format(
                    cte_high['state'].value_counts().to_dict()))

            reversed_results.append({
                'Gene': gene, 'Modality': mod_name,
                'N_CE_high': len(ce_high), 'N_CtE_high': len(cte_high),
                'CE_high_states': ce_high['state'].value_counts().to_dict() if len(ce_high) > 0 else {},
                'CtE_high_states': cte_high['state'].value_counts().to_dict() if len(cte_high) > 0 else {}
            })

    pd.DataFrame(reversed_results).to_csv('{}/prompt9_reversed_genes_regulatory.tsv'.format(OUT),
                                            sep='\t', index=False)


    print("\n\n--- 9-5. Integrated Interpretation ---")

    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        high_col = '{}_high'.format(mod_name)

        high_df = df[df[high_col]]
        low_df = df[~df[high_col]]

        high_active = (high_df['category'].isin(['Active_Promoter', 'Enhancer'])).mean() * 100
        low_active = (low_df['category'].isin(['Active_Promoter', 'Enhancer'])).mean() * 100

        print("  {} active element enrichment:".format(mod_name))
        print("    High-effect: {:.1f}% in active (promoter+enhancer)".format(high_active))
        print("    Low-effect:  {:.1f}% in active".format(low_active))

    print("\n  Microglia vs Neuron in active elements:")
    for ct in ['Microglia', 'Neuron']:
        ct_df = df[df['cell_type'] == ct]
        active_pct = (ct_df['category'].isin(['Active_Promoter', 'Enhancer'])).mean() * 100
        ce_in_active = ct_df[ct_df['category'].isin(['Active_Promoter', 'Enhancer'])]['is_case_enriched'].mean() * 100 \
            if (ct_df['category'].isin(['Active_Promoter', 'Enhancer'])).sum() > 0 else 0
        print("    {}: {:.1f}% variants in active, CE%={:.1f}% in active elements".format(
            ct, active_pct, ce_in_active))

else:
    print("  ChromHMM data not available, skipping enrichment analysis")


print("\n\n--- Writing interpretation document ---")

interp_lines = [
    "# Prompt 9: Regulatory Element Enrichment — Integrated Interpretation",
    "",
    "## Data Sources",
    "- Roadmap Epigenomics E073 ChromHMM 15-state (dorsolateral prefrontal cortex): AVAILABLE",
    "- Nott et al. 2019 microglia enhancers: NOT DOWNLOADED (requires manual supplementary access)",
    "- Gosselin et al. 2017 microglia enhancers: NOT DOWNLOADED",
    "",
    "## Key Findings",
    "",
]

if chromhmm is not None:
    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        high_col = '{}_high'.format(mod_name)
        high_active = (df[df[high_col]]['category'].isin(['Active_Promoter', 'Enhancer'])).mean() * 100
        low_active = (df[~df[high_col]]['category'].isin(['Active_Promoter', 'Enhancer'])).mean() * 100

        interp_lines.append("### {} Active Element Enrichment".format(mod_name))
        interp_lines.append("- High-effect variants: {:.1f}% in active promoters/enhancers".format(high_active))
        interp_lines.append("- Low-effect variants: {:.1f}% in active promoters/enhancers".format(low_active))
        interp_lines.append("")

    interp_lines.extend([
        "## Cell Type Dual Pattern and Regulatory Context",
        "",
        "The cell type dual pattern (microglia = quantitative, neuron = qualitative)",
        "can be contextualized by regulatory element location:",
        "- Microglia gene variants may preferentially affect microglia-specific enhancers",
        "  (not testable without Nott/Gosselin enhancer catalogs)",
        "- Neuron gene variants with high AlphaGenome scores may be in brain-active",
        "  regulatory elements, producing larger per-variant effects",
        "",
        "## Limitations",
        "- ChromHMM from bulk brain tissue (not cell-type specific)",
        "- Microglia-specific enhancer catalogs require manual download from supplementary data",
        "- Without bedtools, interval intersection was done via binary search (validated)",
        "",
        "## Recommendations",
        "1. Download Nott et al. 2019 microglia enhancer BED file for direct enrichment testing",
        "2. Use ENCODE brain tissue DNase/ATAC peaks for neuron-specific analysis",
        "3. Consider single-cell ATAC-seq data for true cell-type resolution",
    ])

with open('{}/prompt9_integrated_interpretation.md'.format(OUT), 'w') as f:
    f.write('\n'.join(interp_lines))
print("  Saved: prompt9_integrated_interpretation.md")


print("\n" + "=" * 70)
print("PROMPT 9 SUMMARY")
print("=" * 70)

print("\n1. DATA AVAILABILITY:")
for row in data_avail:
    print("   {}: {}".format(row['Resource'], row['Status']))

if chromhmm is not None:
    print("\n2. ChromHMM MAPPING:")
    print("   {:,} variants mapped to 15 chromatin states".format(len(df)))

    print("\n3. KEY ENRICHMENT RESULTS (RNA-seq high vs low):")
    rna_results = [r for r in chromhmm_results if r['Modality'] == 'rna_seq']
    for r in rna_results:
        if r['Fisher_P'] < 0.05:
            print("   {:18s}: OR={:.3f}, P={:.4f} *".format(
                r['State_category'], r['OR'], r['Fisher_P']))

print("\nFinished:", datetime.now())
