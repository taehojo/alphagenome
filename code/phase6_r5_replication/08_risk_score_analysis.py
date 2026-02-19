import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '<WORK_DIR>/analysis/r5_replication'
AG_MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
AG_LABELS = {'rna_seq_effect': 'RNA-seq', 'cage_effect': 'CAGE',
             'dnase_effect': 'DNase', 'chip_histone_effect': 'ChIP-histone'}

CELL_TYPES = {
    'Neuron': ['APP', 'PSEN1', 'PSEN2', 'MAPT', 'BIN1', 'CLU', 'SORL1', 'ANK3',
               'PTK2B', 'ADAM10', 'APH1B', 'FERMT2', 'SLC24A4', 'CASS4', 'PICALM',
               'CD2AP', 'EPHA1'],
    'Microglia': ['TREM2', 'PLCG2', 'ABI3', 'INPP5D', 'SPI1', 'CD33', 'MS4A4A',
                  'MS4A6A', 'CR1', 'PILRA', 'LILRB2', 'TREML2', 'SCIMP', 'CLNK', 'BLNK'],
    'Astrocyte': ['CLU', 'APOE', 'ABCA7', 'ABCA1', 'GRN', 'CTSH', 'CTSB'],
    'Ubiquitous': ['ADAM17', 'JAZF1', 'UMAD1', 'RHOH', 'RASGEF1C', 'HS3ST5', 'SNX1',
                   'PLEKHA1', 'WDR81', 'WDR12', 'MINDY2', 'TSPAN14', 'EPDR1', 'NCK2',
                   'TMEM106B', 'SPPL2A', 'EED', 'ACE', 'TPCN1', 'MME', 'ICA1', 'SORT1',
                   'ANKH', 'FOXF1', 'USP6NL', 'IDUA', 'KLF16', 'COX7C', 'SPDYE3',
                   'RBCK1', 'SHARPIN', 'TNIP1', 'TSPOAP1', 'CASP7', 'PRKD3', 'WNT3',
                   'HLA-DQA1', 'MYO15A', 'PRDM7', 'RIN3', 'ADAMTS1', 'IL34', 'DOC2A',
                   'APBB3', 'SIGLEC11', 'BCKDK', 'UNC5CL', 'SEC61G'],
}

def gene_to_celltype(gene):
    for ct, genes in CELL_TYPES.items():
        if gene in genes:
            return ct
    return 'Ubiquitous'


def calc_ir(data, modality, threshold_pct=80):
    mod_data = data[data[modality].notna()].copy()
    if len(mod_data) < 20:
        return None
    threshold = mod_data[modality].quantile(threshold_pct / 100)
    if threshold == 0:
        high = mod_data[mod_data[modality] > 0]
        low = mod_data[mod_data[modality] == 0]
    else:
        high = mod_data[mod_data[modality] >= threshold]
        low = mod_data[mod_data[modality] < threshold]
    if len(high) < 3 or len(low) < 3:
        return None
    high_ce = high['is_case_enriched'].mean() * 100
    low_ce = low['is_case_enriched'].mean() * 100
    ir = high_ce / low_ce if low_ce > 0 else np.inf
    a = int(high['is_case_enriched'].sum())
    b = len(high) - a
    c = int(low['is_case_enriched'].sum())
    d = len(low) - c
    try:
        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])
    except:
        odds_ratio, fisher_p = np.nan, np.nan
    try:
        log_or = np.log(odds_ratio)
        se = np.sqrt(1.0/(a+0.5) + 1.0/(b+0.5) + 1.0/(c+0.5) + 1.0/(d+0.5))
        ci_low = np.exp(log_or - 1.96 * se)
        ci_high = np.exp(log_or + 1.96 * se)
    except:
        ci_low, ci_high = np.nan, np.nan
    return {'IR': ir, 'OR': odds_ratio, 'fisher_p': fisher_p,
            'n_high': len(high), 'n_low': len(low),
            'high_ce_pct': high_ce, 'low_ce_pct': low_ce,
            'OR_CI_low': ci_low, 'OR_CI_high': ci_high}


def calc_gene_ir(data, modality, threshold_pct=80, min_variants=10):
    mod_data = data[data[modality].notna()].copy()
    if len(mod_data) < 20:
        return pd.DataFrame()
    threshold = mod_data[modality].quantile(threshold_pct / 100)
    results = []
    for gene, gdf in mod_data.groupby('gene_name'):
        if len(gdf) < min_variants:
            continue
        if threshold == 0:
            high = gdf[gdf[modality] > 0]
            low = gdf[gdf[modality] == 0]
        else:
            high = gdf[gdf[modality] >= threshold]
            low = gdf[gdf[modality] < threshold]
        if len(high) < 2 or len(low) < 2:
            continue
        high_ce = high['is_case_enriched'].mean() * 100
        low_ce = low['is_case_enriched'].mean() * 100
        ir = high_ce / low_ce if low_ce > 0 else np.inf
        results.append({'gene': gene, 'IR': ir, 'N': len(gdf),
                        'N_high': len(high), 'N_low': len(low)})
    return pd.DataFrame(results)


print("=" * 70)
print("Phase D: Asian Extension Analysis")
print("Started: {}".format(datetime.now()))
print("=" * 70)


print("\n--- D-1. Asian Cohort ---")
samples_all = pd.read_csv('{}/R5_only_samples.tsv'.format(OUT), sep='\t')
asian_samples = samples_all[samples_all['Population'] == 'Asian']

ad_n = len(asian_samples[asian_samples['Diagnosis'] == 'AD'])
cn_n = len(asian_samples[asian_samples['Diagnosis'] == 'CN'])
print("R5-only Asian: {:,} (AD {:,}, CN {:,}, ratio 1:{:.1f})".format(
    len(asian_samples), ad_n, cn_n, cn_n / ad_n))

ad = asian_samples[asian_samples['Diagnosis'] == 'AD']
cn = asian_samples[asian_samples['Diagnosis'] == 'CN']
ad_age = pd.to_numeric(ad['Age'], errors='coerce')
cn_age = pd.to_numeric(cn['Age'], errors='coerce')
print("  Age: AD={:.1f}+/-{:.1f}, CN={:.1f}+/-{:.1f}".format(
    ad_age.mean(), ad_age.std(), cn_age.mean(), cn_age.std()))
ad_apoe = pd.to_numeric(ad['APOE'], errors='coerce')
cn_apoe = pd.to_numeric(cn['APOE'], errors='coerce')
ad_e4 = ad_apoe.isin([34, 44, 24]).sum()
cn_e4 = cn_apoe.isin([34, 44, 24]).sum()
print("  APOE e4+: AD={}/{} ({:.1f}%), CN={}/{} ({:.1f}%)".format(
    ad_e4, ad_n, ad_e4 / ad_n * 100, cn_e4, cn_n, cn_e4 / cn_n * 100))

print("\nR4 Asian: 39 AD, ~2,560 CN (from CLAUDE.md)")
print("  -> R4 Asian was severely underpowered; this is the FIRST adequately powered Asian analysis")

demo = pd.DataFrame([{
    'Cohort': 'R5_Asian', 'AD': ad_n, 'CN': cn_n, 'Total': len(asian_samples),
    'Ratio': cn_n / ad_n,
    'AD_age_mean': ad_age.mean(), 'CN_age_mean': cn_age.mean(),
    'AD_APOE_e4_pct': ad_e4 / ad_n * 100, 'CN_APOE_e4_pct': cn_e4 / cn_n * 100
}, {
    'Cohort': 'R4_Asian', 'AD': 39, 'CN': 2560, 'Total': 2599,
    'Ratio': 2560 / 39
}])
demo.to_csv('{}/Phase_D_asian_demographics.tsv'.format(OUT), sep='\t', index=False)


print("\n--- D-2. Asian Variants ---")

asian_var_file = '{}/R5_only_Asian_variants_AC3.tsv'.format(OUT)
if os.path.exists(asian_var_file):
    asian_vars = pd.read_csv(asian_var_file, sep='\t')
    print("Asian variants (pop-specific MAF<1%, AC>=3): {:,}".format(len(asian_vars)))
else:
    print("ERROR: Asian variant file not found")
    import sys; sys.exit(1)

asian_vars['variant_key'] = asian_vars.apply(
    lambda r: '{}_{}_{}_{}'.format(int(r['chr_num']), int(r['pos']), r['REF'], r['ALT']), axis=1)

asian_vars['total_AC'] = asian_vars['case_AC'] + asian_vars['ctrl_AC']
asian_dedup = asian_vars.sort_values('total_AC', ascending=False).drop_duplicates('variant_key', keep='first')

asian_dedup = asian_dedup.copy()
asian_dedup['cc_ratio'] = asian_dedup['case_AF'] / asian_dedup['ctrl_AF']
asian_dedup = asian_dedup[np.isfinite(asian_dedup['cc_ratio'])].copy()

asian_dedup['enrichment_cat'] = asian_dedup['cc_ratio'].apply(
    lambda x: 'case_enriched' if x > 1 else 'control_enriched')
asian_dedup.loc[(asian_dedup['case_AC'] > 0) & (asian_dedup['ctrl_AC'] == 0), 'enrichment_cat'] = 'case_only'
asian_dedup['is_case_enriched'] = asian_dedup['enrichment_cat'].isin(['case_enriched', 'case_only'])

print("After dedup + finite CC_ratio: {:,}".format(len(asian_dedup)))
print("CE%: {:.1f}%".format(asian_dedup['is_case_enriched'].mean() * 100))

r4_full = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
r4_full['variant_key'] = r4_full.apply(
    lambda r: '{}_{}_{}_{}'.format(int(r['chr_num']), int(r['pos']), r['REF'], r['ALT']), axis=1)
r4_full['variant_key_flip'] = r4_full.apply(
    lambda r: '{}_{}_{}_{}'.format(int(r['chr_num']), int(r['pos']), r['ALT'], r['REF']), axis=1)

ag_map = {}
for _, row in r4_full.iterrows():
    vk = row['variant_key']
    if vk not in ag_map:
        ag_map[vk] = {mod: row[mod] for mod in AG_MODALITIES if mod in row.index and pd.notna(row[mod])}
    vk_flip = row['variant_key_flip']
    if vk_flip not in ag_map:
        ag_map[vk_flip] = {mod: row[mod] for mod in AG_MODALITIES if mod in row.index and pd.notna(row[mod])}

matched = 0
for mod in AG_MODALITIES:
    asian_dedup[mod] = np.nan

for idx, row in asian_dedup.iterrows():
    vk = row['variant_key']
    scores = ag_map.get(vk)
    if scores:
        for mod, val in scores.items():
            asian_dedup.at[idx, mod] = val
        matched += 1

ag_coverage = asian_dedup['rna_seq_effect'].notna().sum()
print("AlphaGenome matched: {:,}/{:,} ({:.1f}%)".format(
    ag_coverage, len(asian_dedup), ag_coverage / len(asian_dedup) * 100))

asian_ag = asian_dedup[asian_dedup['rna_seq_effect'].notna()].copy()
print("Variants with AG scores: {:,}".format(len(asian_ag)))

asian_dedup.to_csv('{}/Phase_D_asian_variants.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- D-3. Asian-Specific Analysis ---")

print("\nTable 2: Mean AG Score (CE vs CtE)")
t2_rows = []
for mod in AG_MODALITIES:
    ce = asian_ag[asian_ag['is_case_enriched'] == True][mod].dropna()
    cte = asian_ag[asian_ag['is_case_enriched'] == False][mod].dropna()
    if len(ce) > 5 and len(cte) > 5:
        _, p = stats.mannwhitneyu(ce, cte, alternative='two-sided')
        pct_diff = (ce.mean() - cte.mean()) / cte.mean() * 100 if cte.mean() != 0 else 0
        d = (ce.mean() - cte.mean()) / np.sqrt((ce.std()**2 + cte.std()**2) / 2) if (ce.std() + cte.std()) > 0 else 0
        t2_rows.append({
            'Modality': AG_LABELS[mod], 'CE_mean': ce.mean(), 'CtE_mean': cte.mean(),
            'pct_diff': pct_diff, 'P': p, 'cohens_d': d, 'CE_higher': ce.mean() > cte.mean()
        })
        print("  {:15s} CE={:.4f} CtE={:.4f} diff={:+.1f}% P={:.2e} CE>CtE={}".format(
            AG_LABELS[mod], ce.mean(), cte.mean(), pct_diff, p, ce.mean() > cte.mean()))

print("\nTable 3: Interaction Ratio (top 20%)")
t3_rows = []
for mod in AG_MODALITIES:
    ir = calc_ir(asian_ag, mod, threshold_pct=80)
    if ir:
        t3_rows.append({'Modality': AG_LABELS[mod], **ir})
        print("  {:15s} IR={:.3f} OR={:.2f} [{:.2f}-{:.2f}] P={:.2e}".format(
            AG_LABELS[mod], ir['IR'], ir['OR'], ir['OR_CI_low'], ir['OR_CI_high'], ir['fisher_p']))

print("\nGene-specific IR (RNA-seq):")
gene_ir = calc_gene_ir(asian_ag, 'rna_seq_effect', threshold_pct=80, min_variants=10)
print("  Genes: {}".format(len(gene_ir)))
if len(gene_ir) > 0:
    print("{:12s} {:>8s} {:>6s}".format('Gene', 'IR', 'N'))
    for _, row in gene_ir.sort_values('IR', ascending=False).head(10).iterrows():
        print("{:12s} {:>8.3f} {:>6d}".format(row['gene'], row['IR'], row['N']))

print("\nCell Type Pattern:")
asian_ag['cell_type'] = asian_ag['gene_name'].apply(gene_to_celltype)
for ct in ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']:
    ct_data = asian_ag[asian_ag['cell_type'] == ct]
    if len(ct_data) > 0:
        print("  {:12s} N={:>5d} CC_ratio={:.3f} RNA-seq={:.4f} CE%={:.1f}%".format(
            ct, len(ct_data), ct_data['cc_ratio'].mean(),
            ct_data['rna_seq_effect'].mean(), ct_data['is_case_enriched'].mean() * 100))


print("\n\n--- D-4. Cross-Population Comparison ---")

r4_gene_ir = pd.read_csv('<WORK_DIR>/results/final_valid_audit_20260202/all_gene_ir.csv')

print("\n{:15s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
    'Metric', 'R4 all', 'R5 NHW+H+AA', 'R5 Asian', 'R5 NHW'))
print("-" * 70)

phase_b_t3 = pd.read_csv('{}/Phase_B_table3_replication.tsv'.format(OUT), sep='\t')
phase_c_summary = pd.read_csv('{}/Phase_C_bootstrap_IR_summary.tsv'.format(OUT), sep='\t')

r4_ir_map = {'RNA-seq': 1.086, 'CAGE': 1.056, 'DNase': 1.028, 'ChIP-histone': 1.062}

for mod in AG_MODALITIES:
    label = AG_LABELS[mod]
    r4_val = r4_ir_map.get(label, np.nan)

    b3_row = phase_b_t3[phase_b_t3['Modality'] == label]
    r5_rep = b3_row['R5_IR'].values[0] if len(b3_row) > 0 else np.nan

    asian_ir = calc_ir(asian_ag, mod, threshold_pct=80)
    r5_asian = asian_ir['IR'] if asian_ir else np.nan

    pop_df = pd.read_csv('{}/Phase_B_by_population.tsv'.format(OUT), sep='\t')
    nhw_row = pop_df[pop_df['Population'] == 'NHW']
    r5_nhw = nhw_row['{}_IR'.format(label)].values[0] if len(nhw_row) > 0 and '{}_IR'.format(label) in nhw_row.columns else np.nan

    print("{:15s} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(
        label + ' IR', r4_val, r5_rep, r5_asian, r5_nhw))

if len(gene_ir) > 0 and len(r4_gene_ir) > 0:
    m = gene_ir.merge(r4_gene_ir[['gene', 'IR']], on='gene', how='inner', suffixes=('_Asian', '_R4'))
    if len(m) > 2:
        r, p = stats.spearmanr(m['IR_R4'], m['IR_Asian'])
        print("\nAsian gene IR vs R4: {} genes, Spearman r={:.3f}, P={:.2e}".format(len(m), r, p))
        m['same_dir'] = ((m['IR_R4'] > 1) & (m['IR_Asian'] > 1)) | ((m['IR_R4'] < 1) & (m['IR_Asian'] < 1))
        print("Direction concordance: {}/{} ({:.1f}%)".format(
            m['same_dir'].sum(), len(m), m['same_dir'].mean() * 100))


results_rows = []
for mod in AG_MODALITIES:
    label = AG_LABELS[mod]
    ir = calc_ir(asian_ag, mod, threshold_pct=80)
    if ir:
        results_rows.append({
            'Modality': label,
            'Asian_IR': ir['IR'], 'Asian_OR': ir['OR'],
            'Asian_OR_CI': '[{:.2f}-{:.2f}]'.format(ir['OR_CI_low'], ir['OR_CI_high']),
            'Asian_P': ir['fisher_p'],
            'Asian_N_high': ir['n_high'], 'Asian_N_low': ir['n_low'],
            'R4_IR': r4_ir_map.get(label, np.nan),
            'Same_direction': (ir['IR'] >= 1) == (r4_ir_map.get(label, 1) >= 1)
        })

pd.DataFrame(results_rows).to_csv('{}/Phase_D_asian_results.tsv'.format(OUT), sep='\t', index=False)

if len(gene_ir) > 0:
    gene_ir.to_csv('{}/Phase_D_asian_gene_IR.tsv'.format(OUT), sep='\t', index=False)

comp_rows = []
for mod in AG_MODALITIES:
    label = AG_LABELS[mod]
    asian_ir = calc_ir(asian_ag, mod, threshold_pct=80)
    pop_df = pd.read_csv('{}/Phase_B_by_population.tsv'.format(OUT), sep='\t')
    comp_rows.append({
        'Modality': label,
        'R4_all_IR': r4_ir_map.get(label, np.nan),
        'R5_NHW_IR': pop_df[pop_df['Population'] == 'NHW']['{}_IR'.format(label)].values[0] if len(pop_df[pop_df['Population'] == 'NHW']) > 0 and '{}_IR'.format(label) in pop_df.columns else np.nan,
        'R5_Hispanic_IR': pop_df[pop_df['Population'] == 'Hispanic']['{}_IR'.format(label)].values[0] if len(pop_df[pop_df['Population'] == 'Hispanic']) > 0 and '{}_IR'.format(label) in pop_df.columns else np.nan,
        'R5_AA_IR': pop_df[pop_df['Population'] == 'AA']['{}_IR'.format(label)].values[0] if len(pop_df[pop_df['Population'] == 'AA']) > 0 and '{}_IR'.format(label) in pop_df.columns else np.nan,
        'R5_Asian_IR': asian_ir['IR'] if asian_ir else np.nan
    })
pd.DataFrame(comp_rows).to_csv('{}/Phase_D_asian_vs_R4_comparison.tsv'.format(OUT), sep='\t', index=False)


print("\n" + "=" * 70)
print("PHASE D SUMMARY")
print("=" * 70)
print("\nAsian cohort: {} AD, {} CN (ratio 1:{:.1f})".format(ad_n, cn_n, cn_n / ad_n))
print("Variants with AG: {:,}".format(len(asian_ag)))
ce_pct = asian_ag['is_case_enriched'].mean() * 100
print("CE%: {:.1f}%".format(ce_pct))
print("\nIR (top 20%/bottom 80%):")
for row in results_rows:
    print("  {:15s} IR={:.3f} P={:.2e} {}".format(
        row['Modality'], row['Asian_IR'], row['Asian_P'],
        'SAME as R4' if row['Same_direction'] else 'OPPOSITE'))

print("\nFinished: {}".format(datetime.now()))
