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
REPLICATION_POPS = ['NHW', 'Hispanic', 'AA']

CELL_TYPES = {
    'Neuron': ['APP', 'PSEN1', 'PSEN2', 'MAPT', 'BIN1', 'CLU', 'SORL1', 'ANK3',
               'PTK2B', 'ADAM10', 'APH1B', 'FERMT2', 'SLC24A4', 'CASS4', 'PICALM',
               'CD2AP', 'EPHA1'],
    'Microglia': ['TREM2', 'PLCG2', 'ABI3', 'INPP5D', 'SPI1', 'CD33', 'MS4A4A',
                  'MS4A6A', 'CR1', 'PILRA', 'LILRB2', 'TREML2', 'SCIMP', 'CLNK',
                  'BLNK'],
    'Astrocyte': ['CLU', 'APOE', 'ABCA7', 'ABCA1', 'GRN', 'CTSH', 'CTSB'],
    'Ubiquitous': ['ADAM17', 'JAZF1', 'UMAD1', 'RHOH', 'RASGEF1C', 'HS3ST5', 'SNX1',
                   'PLEKHA1', 'WDR81', 'WDR12', 'MINDY2', 'TSPAN14', 'EPDR1', 'NCK2',
                   'TMEM106B', 'SPPL2A', 'EED', 'ACE', 'TPCN1', 'MME', 'ICA1', 'SORT1',
                   'ANKH', 'FOXF1', 'USP6NL', 'IDUA', 'KLF16', 'COX7C', 'SPDYE3',
                   'RBCK1', 'SHARPIN', 'TNIP1', 'TSPOAP1', 'CASP7', 'PRKD3', 'WNT3',
                   'HLA-DQA1', 'MYO15A', 'PRDM7', 'RIN3', 'ADAMTS1', 'IL34', 'DOC2A',
                   'APBB3', 'SIGLEC11', 'BCKDK', 'UNC5CL', 'SEC61G'],
}

R4_REF = {
    'RNA-seq': {'IR': 1.086, 'OR': 1.36},
    'CAGE': {'IR': 1.056},
    'DNase': {'IR': 1.028},
    'ChIP-histone': {'IR': 1.062}
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
                        'N_high': len(high), 'N_low': len(low),
                        'high_ce_pct': high_ce, 'low_ce_pct': low_ce})
    return pd.DataFrame(results)


def run_full_analysis(data, label, r4_dedup):
    results = {'label': label, 'N': len(data)}
    ce_pct = data['is_case_enriched'].mean() * 100
    results['CE_pct'] = ce_pct

    t2_rows = []
    for mod in AG_MODALITIES:
        ce = data[data['is_case_enriched'] == True][mod].dropna()
        cte = data[data['is_case_enriched'] == False][mod].dropna()
        if len(ce) > 5 and len(cte) > 5:
            _, p = stats.mannwhitneyu(ce, cte, alternative='two-sided')
            d = (ce.mean() - cte.mean()) / np.sqrt((ce.std()**2 + cte.std()**2) / 2) if (ce.std() + cte.std()) > 0 else 0
            pct_diff = (ce.mean() - cte.mean()) / cte.mean() * 100 if cte.mean() != 0 else 0
            t2_rows.append({
                'Modality': AG_LABELS[mod], 'CE_mean': ce.mean(), 'CtE_mean': cte.mean(),
                'pct_diff': pct_diff, 'P': p, 'cohens_d': d, 'N_CE': len(ce), 'N_CtE': len(cte)
            })
    results['table2'] = pd.DataFrame(t2_rows)

    t3_rows = []
    for mod in AG_MODALITIES:
        ir = calc_ir(data, mod, threshold_pct=80)
        if ir:
            t3_rows.append({
                'Modality': AG_LABELS[mod], **ir
            })
    results['table3'] = pd.DataFrame(t3_rows)

    gene_ir = calc_gene_ir(data, 'rna_seq_effect', threshold_pct=80, min_variants=10)
    results['gene_ir'] = gene_ir

    data_ct = data.copy()
    data_ct['cell_type'] = data_ct['gene_name'].apply(gene_to_celltype)
    ct_rows = []
    for ct in ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']:
        ct_data = data_ct[data_ct['cell_type'] == ct]
        if len(ct_data) > 0:
            ct_rows.append({
                'cell_type': ct,
                'N': len(ct_data),
                'mean_cc_ratio': ct_data['cc_ratio'].mean() if 'cc_ratio' in ct_data.columns else np.nan,
                'mean_rna_seq': ct_data['rna_seq_effect'].mean() if 'rna_seq_effect' in ct_data.columns else np.nan,
                'mean_cage': ct_data['cage_effect'].mean() if 'cage_effect' in ct_data.columns else np.nan,
                'CE_pct': ct_data['is_case_enriched'].mean() * 100
            })
    results['celltype'] = pd.DataFrame(ct_rows)

    return results


print("=" * 70)
print("Phase B: Shared Variant Replication (NHW + Hispanic + AA)")
print("Started: {}".format(datetime.now()))
print("=" * 70)

r5_variants = pd.read_csv('{}/Phase_A_replication_variants_filtered.tsv'.format(OUT), sep='\t')
r4_full = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')

r4_full['total_AC'] = r4_full['case_AC'] + r4_full['ctrl_AC']
r4_dedup = r4_full[r4_full['total_AC'] >= 3].sort_values('total_AC', ascending=False) \
                  .drop_duplicates('variant_id', keep='first')

r4_dedup['variant_key'] = r4_dedup.apply(
    lambda r: '{}_{}_{}_{}'.format(int(r['chr_num']), int(r['pos']), r['REF'], r['ALT']), axis=1)
r4_dedup['variant_key_flip'] = r4_dedup.apply(
    lambda r: '{}_{}_{}_{}'.format(int(r['chr_num']), int(r['pos']), r['ALT'], r['REF']), axis=1)

print("R4 dedup: {:,} variants".format(len(r4_dedup)))
print("R5 replication: {:,} variants".format(len(r5_variants)))


print("\n--- B-1. Shared Variant Identification ---")

r4_keys = set(r4_dedup['variant_key'].values)
r4_flip_keys = set(r4_dedup['variant_key_flip'].values)

r5_keys = set(r5_variants['variant_key'].values)

shared_direct = r5_keys & r4_keys
shared_flipped = r5_keys & r4_flip_keys
shared_all = shared_direct | shared_flipped

print("R4 variants: {:,}".format(len(r4_keys)))
print("R5 rep variants: {:,}".format(len(r5_keys)))
print("Shared (direct match): {:,}".format(len(shared_direct)))
print("Shared (allele flip): {:,}".format(len(shared_flipped - shared_direct)))
print("Shared total: {:,}".format(len(shared_all)))
print("R4-only: {:,}".format(len(r4_keys - r5_keys - r4_flip_keys)))
print("R5-only: {:,}".format(len(r5_keys - r4_keys - r4_flip_keys)))

shared_rows = []
r4_vk_map = r4_dedup.set_index('variant_key')
r4_flip_map = r4_dedup.set_index('variant_key_flip')

for _, r5_row in r5_variants.iterrows():
    vk = r5_row['variant_key']
    r4_row = None
    if vk in r4_vk_map.index:
        r4_row = r4_vk_map.loc[vk]
    elif vk in r4_flip_map.index:
        r4_row = r4_flip_map.loc[vk]
    else:
        continue

    if isinstance(r4_row, pd.DataFrame):
        r4_row = r4_row.iloc[0]

    row = {
        'variant_key': vk,
        'chr_num': int(r5_row['chr_num']),
        'pos': int(r5_row['pos']),
        'gene_name': r5_row['gene_name'],
        'source_pop': r5_row['source_pop'],
        'R4_cc_ratio': r4_row['cc_ratio'] if 'cc_ratio' in r4_row.index else np.nan,
        'R4_enrichment': r4_row['enrichment'] if 'enrichment' in r4_row.index else '',
        'R5_case_AF': r5_row['case_AF'],
        'R5_ctrl_AF': r5_row['ctrl_AF'],
        'R5_case_AC': r5_row['case_AC'],
        'R5_ctrl_AC': r5_row['ctrl_AC'],
        'R5_cc_ratio': r5_row['cc_ratio_recalc'] if 'cc_ratio_recalc' in r5_row.index else r5_row.get('cc_ratio', np.nan),
        'R5_enrichment': r5_row['enrichment'],
    }
    for mod in AG_MODALITIES:
        row[mod] = r4_row[mod] if mod in r4_row.index else np.nan

    shared_rows.append(row)

shared_df = pd.DataFrame(shared_rows)
shared_df['R4_is_CE'] = shared_df['R4_enrichment'].isin(['case_enriched', 'case_only'])
shared_df['R5_is_CE'] = shared_df['R5_enrichment'].isin(['case_enriched', 'case_only'])
shared_df['is_case_enriched'] = shared_df['R5_is_CE']
shared_df['cc_ratio'] = shared_df['R5_cc_ratio']

print("\nShared variants with full data: {:,}".format(len(shared_df)))

ag_coverage = {}
for mod in AG_MODALITIES:
    n_with = shared_df[mod].notna().sum()
    ag_coverage[mod] = n_with
    print("  {} coverage: {:,}/{:,} ({:.1f}%)".format(
        AG_LABELS[mod], n_with, len(shared_df), n_with / len(shared_df) * 100))

shared_df.to_csv('{}/Phase_B_shared_variants.tsv'.format(OUT), sep='\t', index=False)


print("\n--- B-3a. Variant-level Concordance ---")

valid = shared_df[shared_df['R4_cc_ratio'].notna() & shared_df['R5_cc_ratio'].notna() &
                  np.isfinite(shared_df['R4_cc_ratio']) & np.isfinite(shared_df['R5_cc_ratio'])]

r_cc, p_cc = stats.spearmanr(valid['R4_cc_ratio'], valid['R5_cc_ratio'])
print("CC_ratio Spearman: r={:.3f}, P={:.2e} (N={:,})".format(r_cc, p_cc, len(valid)))

concordant = (valid['R4_is_CE'] == valid['R5_is_CE']).sum()
concordance = concordant / len(valid) * 100
print("Enrichment concordance: {:,}/{:,} ({:.1f}%)".format(concordant, len(valid), concordance))

print("\nPer-population concordance:")
conc_rows = [{'Population': 'ALL', 'N': len(valid), 'CC_ratio_r': r_cc,
              'CC_ratio_P': p_cc, 'Concordance_pct': concordance}]
for pop in REPLICATION_POPS:
    pv = valid[valid['source_pop'] == pop]
    if len(pv) > 10:
        r, p = stats.spearmanr(pv['R4_cc_ratio'], pv['R5_cc_ratio'])
        conc = (pv['R4_is_CE'] == pv['R5_is_CE']).mean() * 100
        print("  {}: r={:.3f}, P={:.2e}, concordance={:.1f}% (N={:,})".format(pop, r, p, conc, len(pv)))
        conc_rows.append({'Population': pop, 'N': len(pv), 'CC_ratio_r': r,
                          'CC_ratio_P': p, 'Concordance_pct': conc})
pd.DataFrame(conc_rows).to_csv('{}/Phase_B_concordance.tsv'.format(OUT), sep='\t', index=False)


print("\n--- B-3b. Table 2 Replication ---")

t2_all = []
for mod in AG_MODALITIES:
    ce = shared_df[shared_df['R5_is_CE'] == True][mod].dropna()
    cte = shared_df[shared_df['R5_is_CE'] == False][mod].dropna()
    if len(ce) > 5 and len(cte) > 5:
        _, p = stats.mannwhitneyu(ce, cte, alternative='two-sided')
        pct_diff = (ce.mean() - cte.mean()) / cte.mean() * 100 if cte.mean() != 0 else 0
        d = (ce.mean() - cte.mean()) / np.sqrt((ce.std()**2 + cte.std()**2) / 2) if (ce.std() + cte.std()) > 0 else 0
        t2_all.append({
            'Modality': AG_LABELS[mod],
            'R5_CE_mean': ce.mean(), 'R5_CtE_mean': cte.mean(),
            'R5_pct_diff': pct_diff, 'R5_P': p, 'R5_cohens_d': d,
            'R5_N_CE': len(ce), 'R5_N_CtE': len(cte),
            'R5_CE_higher': ce.mean() > cte.mean()
        })
        print("  {:15s} CE={:.4f} CtE={:.4f} diff={:+.1f}% P={:.2e} d={:.4f} CE>CtE={}".format(
            AG_LABELS[mod], ce.mean(), cte.mean(), pct_diff, p, d, ce.mean() > cte.mean()))

pd.DataFrame(t2_all).to_csv('{}/Phase_B_table2_replication.tsv'.format(OUT), sep='\t', index=False)


print("\n--- B-3c. Table 3 Replication (top 20% threshold) ---")

t3_all = []
for mod in AG_MODALITIES:
    ir = calc_ir(shared_df, mod, threshold_pct=80)
    if ir:
        r4_ir = R4_REF.get(AG_LABELS[mod], {}).get('IR', np.nan)
        same_dir = (ir['IR'] >= 1 and r4_ir >= 1) or (ir['IR'] < 1 and r4_ir < 1)
        t3_all.append({
            'Modality': AG_LABELS[mod],
            'R5_IR': ir['IR'], 'R5_OR': ir['OR'],
            'R5_OR_CI': '[{:.2f}-{:.2f}]'.format(ir['OR_CI_low'], ir['OR_CI_high']),
            'R5_Fisher_P': ir['fisher_p'],
            'R5_N_high': ir['n_high'], 'R5_N_low': ir['n_low'],
            'R4_IR': r4_ir,
            'Same_direction': same_dir
        })
        print("  {:15s} IR={:.3f} OR={:.2f} [{:.2f}-{:.2f}] P={:.2e} (R4 IR={:.3f}) dir={}".format(
            AG_LABELS[mod], ir['IR'], ir['OR'], ir['OR_CI_low'], ir['OR_CI_high'],
            ir['fisher_p'], r4_ir, 'SAME' if same_dir else 'OPPOSITE'))

pd.DataFrame(t3_all).to_csv('{}/Phase_B_table3_replication.tsv'.format(OUT), sep='\t', index=False)


print("\n--- B-3d. Gene-specific IR ---")

r5_gene_ir = calc_gene_ir(shared_df, 'rna_seq_effect', threshold_pct=80, min_variants=10)
print("R5 genes with IR: {}".format(len(r5_gene_ir)))

r4_gene_ir = pd.read_csv('<WORK_DIR>/results/final_valid_audit_20260202/all_gene_ir.csv')
print("R4 genes with IR: {}".format(len(r4_gene_ir)))

if len(r5_gene_ir) > 0:
    merged = r5_gene_ir.merge(r4_gene_ir[['gene', 'IR']], left_on='gene', right_on='gene',
                               how='inner', suffixes=('_R5', '_R4'))
    if len(merged) > 1:
        r, p = stats.spearmanr(merged['IR_R4'], merged['IR_R5'])
        print("Shared genes: {}".format(len(merged)))
        print("Spearman (R4 vs R5 gene IR): r={:.3f}, P={:.2e}".format(r, p))

        merged['same_dir'] = ((merged['IR_R4'] > 1) & (merged['IR_R5'] > 1)) | \
                             ((merged['IR_R4'] < 1) & (merged['IR_R5'] < 1))
        print("Direction concordance: {}/{} ({:.1f}%)".format(
            merged['same_dir'].sum(), len(merged), merged['same_dir'].mean() * 100))

        print("\nKey genes (sorted by R4 IR):")
        print("{:12s} {:>8s} {:>8s} {:>8s}".format('Gene', 'R4_IR', 'R5_IR', 'Direction'))
        for _, row in merged.sort_values('IR_R4', ascending=False).head(10).iterrows():
            d = 'SAME' if row['same_dir'] else 'OPPOSITE'
            print("{:12s} {:>8.3f} {:>8.3f} {:>8s}".format(row['gene'], row['IR_R4'], row['IR_R5'], d))

        for gene in ['APH1B', 'CASP7', 'CD2AP', 'SIGLEC11', 'TREML2', 'LILRB2']:
            g = merged[merged['gene'] == gene]
            if len(g) > 0:
                row = g.iloc[0]
                d = 'SAME' if row['same_dir'] else 'OPPOSITE'
                print("  {} - R4: {:.3f}, R5: {:.3f} ({})".format(gene, row['IR_R4'], row['IR_R5'], d))
            else:
                r5_only = r5_gene_ir[r5_gene_ir['gene'] == gene]
                if len(r5_only) > 0:
                    print("  {} - R4: N/A, R5: {:.3f}".format(gene, r5_only.iloc[0]['IR']))
                else:
                    print("  {} - not in R5 gene IR (insufficient variants)".format(gene))

        merged.to_csv('{}/Phase_B_gene_IR_replication.tsv'.format(OUT), sep='\t', index=False)


print("\n--- B-3e. Cell Type Pattern ---")

shared_df['cell_type'] = shared_df['gene_name'].apply(gene_to_celltype)

ct_rows = []
for ct in ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']:
    ct_data = shared_df[shared_df['cell_type'] == ct]
    if len(ct_data) > 0:
        row = {
            'cell_type': ct,
            'N': len(ct_data),
            'mean_cc_ratio': ct_data['R5_cc_ratio'].mean(),
            'mean_rna_seq': ct_data['rna_seq_effect'].mean(),
            'mean_cage': ct_data['cage_effect'].mean(),
            'CE_pct': ct_data['R5_is_CE'].mean() * 100
        }
        ct_rows.append(row)
        print("  {:12s} N={:>5d} CC_ratio={:.3f} RNA-seq={:.4f} CE%={:.1f}%".format(
            ct, len(ct_data), row['mean_cc_ratio'], row['mean_rna_seq'], row['CE_pct']))

r4_dedup['cell_type'] = r4_dedup['gene_name'].apply(gene_to_celltype)
print("\n  R4 reference:")
for ct in ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']:
    r4_ct = r4_dedup[r4_dedup['cell_type'] == ct]
    if len(r4_ct) > 0:
        print("  {:12s} N={:>5d} CC_ratio={:.3f} RNA-seq={:.4f} CE%={:.1f}%".format(
            ct, len(r4_ct), r4_ct['cc_ratio'].mean(),
            r4_ct['rna_seq_effect'].mean() if 'rna_seq_effect' in r4_ct.columns else 0,
            r4_ct['enrichment'].isin(['case_enriched', 'case_only']).mean() * 100))

print("\n  Dual pattern check:")
print("  R4: Microglia=high CC_ratio+low per-variant effect, Neuron=low CC_ratio+high per-variant effect")
if len(ct_rows) >= 2:
    micro = [r for r in ct_rows if r['cell_type'] == 'Microglia']
    neuro = [r for r in ct_rows if r['cell_type'] == 'Neuron']
    if micro and neuro:
        micro_cc = micro[0]['mean_cc_ratio']
        neuro_cc = neuro[0]['mean_cc_ratio']
        micro_rna = micro[0]['mean_rna_seq']
        neuro_rna = neuro[0]['mean_rna_seq']
        dual_cc = micro_cc > neuro_cc
        dual_rna = micro_rna < neuro_rna
        print("  R5: Microglia CC > Neuron CC? {} ({:.3f} vs {:.3f})".format(dual_cc, micro_cc, neuro_cc))
        print("  R5: Microglia RNA < Neuron RNA? {} ({:.4f} vs {:.4f})".format(dual_rna, micro_rna, neuro_rna))
        print("  Dual pattern replicated: {}".format(dual_cc and dual_rna))

pd.DataFrame(ct_rows).to_csv('{}/Phase_B_celltype_replication.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- B-3f. Per-Population Analysis ---")

pop_results = []

for pop in REPLICATION_POPS:
    pop_data = shared_df[shared_df['source_pop'] == pop].copy()
    if len(pop_data) < 50:
        print("\n{}: only {:,} shared variants, skipping detailed analysis".format(pop, len(pop_data)))
        pop_results.append({
            'Population': pop, 'N': len(pop_data),
            'CE_pct': pop_data['R5_is_CE'].mean() * 100 if len(pop_data) > 0 else np.nan
        })
        continue

    print("\n=== {} (N={:,}) ===".format(pop, len(pop_data)))
    ce_pct = pop_data['R5_is_CE'].mean() * 100
    print("  CE%: {:.1f}%".format(ce_pct))

    pop_row = {'Population': pop, 'N': len(pop_data), 'CE_pct': ce_pct}

    for mod in AG_MODALITIES:
        ir = calc_ir(pop_data, mod, threshold_pct=80)
        if ir:
            pop_row['{}_IR'.format(AG_LABELS[mod])] = ir['IR']
            pop_row['{}_P'.format(AG_LABELS[mod])] = ir['fisher_p']
            print("  {:15s} IR={:.3f} P={:.2e}".format(AG_LABELS[mod], ir['IR'], ir['fisher_p']))
        else:
            pop_row['{}_IR'.format(AG_LABELS[mod])] = np.nan

    pop_gene_ir = calc_gene_ir(pop_data, 'rna_seq_effect', threshold_pct=80, min_variants=10)
    pop_row['N_genes_IR'] = len(pop_gene_ir)
    if len(pop_gene_ir) > 0 and len(r4_gene_ir) > 0:
        m = pop_gene_ir.merge(r4_gene_ir[['gene', 'IR']], on='gene', how='inner', suffixes=('_R5', '_R4'))
        if len(m) > 2:
            r, p = stats.spearmanr(m['IR_R4'], m['IR_R5'])
            pop_row['Gene_IR_r'] = r
            pop_row['Gene_IR_p'] = p
            print("  Gene IR: {} genes, Spearman r={:.3f}, P={:.2e}".format(len(m), r, p))

    pop_results.append(pop_row)

pop_df = pd.DataFrame(pop_results)
print("\n\n--- Per-Population Summary ---")
print(pop_df.to_string(index=False))
pop_df.to_csv('{}/Phase_B_by_population.tsv'.format(OUT), sep='\t', index=False)


print("\n" + "=" * 70)
print("PHASE B SUMMARY")
print("=" * 70)
print("\nShared variants: {:,}/{:,} R4 ({:.1f}%)".format(
    len(shared_df), len(r4_dedup), len(shared_df) / len(r4_dedup) * 100))
print("CC_ratio Spearman: r={:.3f}".format(r_cc))
print("Enrichment concordance: {:.1f}%".format(concordance))

print("\nTable 3 (IR, top 20%/bottom 80%):")
for row in t3_all:
    print("  {:15s} R5 IR={:.3f} (R4={:.3f}) {}".format(
        row['Modality'], row['R5_IR'], row['R4_IR'],
        'SAME' if row['Same_direction'] else 'OPPOSITE'))

if len(r5_gene_ir) > 0 and 'merged' in dir() and len(merged) > 1:
    print("\nGene IR correlation: r={:.3f} (P={:.2e})".format(r, p))

print("\nFinished: {}".format(datetime.now()))
