import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

OUT_DIR = '<WORK_DIR>/analysis/r5_replication'
R5_AG_DIR = '<ADSP_AI_DIR>/R5/resilience_variants/alphagenome'
R4_PKL_DIR = '<ADSP_AI_DIR>/LD-rarevariant-5th-all/worker_results'

AG_MODALITIES = [
    'rna_seq_effect', 'cage_effect', 'procap_effect',
    'splice_sites_effect', 'splice_site_usage_effect', 'splice_junctions_effect',
    'atac_effect', 'dnase_effect', 'chip_histone_effect',
    'chip_tf_effect', 'contact_maps_effect'
]

print(f"{'='*70}")
print(f"Prompt 3: AlphaGenome Score Matching for R5-only Variants")
print(f"Started: {datetime.now()}")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("Step 1: Loading R5-only variants")
print(f"{'='*70}")

r5v = pd.read_csv(f'{OUT_DIR}/R5_only_variants_AC3.tsv', sep='\t')
print(f"R5-only variants: {len(r5v):,}")

r5v['match_key'] = ('chr' + r5v['chr_num'].astype(str) + ':' +
                     r5v['pos'].astype(int).astype(str) + ':' +
                     r5v['REF'] + ':' + r5v['ALT'])
r5v['match_key_flip'] = ('chr' + r5v['chr_num'].astype(str) + ':' +
                          r5v['pos'].astype(int).astype(str) + ':' +
                          r5v['ALT'] + ':' + r5v['REF'])
r5v['pos_key'] = 'chr' + r5v['chr_num'].astype(str) + ':' + r5v['pos'].astype(int).astype(str)

print(f"\n{'='*70}")
print("Step 2: Loading R5 AlphaGenome scores")
print(f"{'='*70}")

ag_r5 = pd.read_csv(f'{R5_AG_DIR}/ag_scores_matched_to_bim.csv')
print(f"R5 AG scores: {len(ag_r5):,} variants")

r5_ag_lookup = {}
for _, row in ag_r5.iterrows():
    vid = row['variant_id']
    scores = {mod: row[mod] for mod in AG_MODALITIES}
    r5_ag_lookup[vid] = scores

print(f"R5 AG lookup entries: {len(r5_ag_lookup):,}")

print(f"\n{'='*70}")
print("Step 3: Loading R4 AlphaGenome pickle data")
print(f"{'='*70}")

r4_ag_lookup = {}
for i in range(5):
    pkl_file = f'{R4_PKL_DIR}/results_00{i}.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        for rec in data:
            vid = rec.get('variant_id', '')
            if vid and vid not in r4_ag_lookup:
                scores = {}
                for mod in AG_MODALITIES:
                    scores[mod] = rec.get(mod, np.nan)
                r4_ag_lookup[vid] = scores
        print(f"  Worker {i}: {len(data):,} records loaded")

print(f"R4 AG unique lookup entries: {len(r4_ag_lookup):,}")

print(f"\n{'='*70}")
print("Step 4: Matching variants to AlphaGenome scores")
print(f"{'='*70}")

matched_scores = []
match_sources = []

for idx, row in r5v.iterrows():
    key = row['match_key']
    key_flip = row['match_key_flip']

    if key in r5_ag_lookup:
        matched_scores.append(r5_ag_lookup[key])
        match_sources.append('r5_ag_direct')
        continue

    if key_flip in r5_ag_lookup:
        matched_scores.append(r5_ag_lookup[key_flip])
        match_sources.append('r5_ag_flipped')
        continue

    if key in r4_ag_lookup:
        matched_scores.append(r4_ag_lookup[key])
        match_sources.append('r4_ag_direct')
        continue

    if key_flip in r4_ag_lookup:
        matched_scores.append(r4_ag_lookup[key_flip])
        match_sources.append('r4_ag_flipped')
        continue

    matched_scores.append({mod: np.nan for mod in AG_MODALITIES})
    match_sources.append('unmatched')

for mod in AG_MODALITIES:
    r5v[mod] = [s[mod] for s in matched_scores]
r5v['ag_match_source'] = match_sources

match_counts = pd.Series(match_sources).value_counts()
print(f"\nMatch statistics:")
for source, count in match_counts.items():
    print(f"  {source:20s}: {count:>8,} ({count/len(r5v)*100:.1f}%)")

total_matched = len(r5v) - (r5v['ag_match_source'] == 'unmatched').sum()
print(f"\nTotal matched: {total_matched:,} / {len(r5v):,} ({total_matched/len(r5v)*100:.1f}%)")
print(f"Unmatched: {(r5v['ag_match_source']=='unmatched').sum():,} ({(r5v['ag_match_source']=='unmatched').sum()/len(r5v)*100:.1f}%)")

print(f"\n{'='*70}")
print("Step 5: AlphaGenome score analysis (matched variants only)")
print(f"{'='*70}")

matched = r5v[r5v['ag_match_source'] != 'unmatched'].copy()
print(f"\nMatched variants for analysis: {len(matched):,}")

case_enr = matched[matched['enrichment'] == 'case_enriched']
ctrl_enr = matched[matched['enrichment'] == 'ctrl_enriched']

print(f"  Case-enriched: {len(case_enr):,}")
print(f"  Ctrl-enriched: {len(ctrl_enr):,}")

print(f"\n{'Modality':30s} {'Case-E Mean':>12s} {'Ctrl-E Mean':>12s} {'Ratio':>8s}")
print("-" * 68)

from scipy import stats

modality_results = []
for mod in AG_MODALITIES:
    case_vals = case_enr[mod].dropna()
    ctrl_vals = ctrl_enr[mod].dropna()

    if len(case_vals) > 10 and len(ctrl_vals) > 10:
        case_mean = case_vals.mean()
        ctrl_mean = ctrl_vals.mean()
        ratio = case_mean / ctrl_mean if ctrl_mean > 0 else np.inf

        try:
            stat, pval = stats.mannwhitneyu(case_vals, ctrl_vals, alternative='two-sided')
        except:
            pval = np.nan

        print(f"{mod:30s} {case_mean:>12.4f} {ctrl_mean:>12.4f} {ratio:>7.3f}  p={pval:.2e}")
        modality_results.append({
            'modality': mod,
            'case_enriched_mean': case_mean,
            'ctrl_enriched_mean': ctrl_mean,
            'ratio': ratio,
            'pvalue': pval,
            'n_case': len(case_vals),
            'n_ctrl': len(ctrl_vals)
        })

mod_df = pd.DataFrame(modality_results)
mod_df.to_csv(f'{OUT_DIR}/R5_only_AG_modality_comparison.tsv', sep='\t', index=False)

print(f"\n{'='*70}")
print("Step 6: Gene-level AlphaGenome analysis")
print(f"{'='*70}")

gene_ag = matched.groupby('gene_name').agg(
    n_variants=('variant_id', 'count'),
    n_case_enriched=('enrichment', lambda x: (x == 'case_enriched').sum()),
    mean_rna_seq=('rna_seq_effect', 'mean'),
    mean_cage=('cage_effect', 'mean'),
    mean_dnase=('dnase_effect', 'mean'),
    mean_chip_histone=('chip_histone_effect', 'mean')
).reset_index()

gene_ag['pct_ce'] = (gene_ag['n_case_enriched'] / gene_ag['n_variants'] * 100).round(1)
gene_ag = gene_ag.sort_values('n_variants', ascending=False)

print(f"\n{'Gene':15s} {'N_var':>6s} {'%CE':>6s} {'RNA-seq':>8s} {'CAGE':>8s} {'DNase':>8s} {'ChIP-H':>8s}")
print("-" * 60)
for _, g in gene_ag.head(20).iterrows():
    print(f"{g['gene_name']:15s} {g['n_variants']:>6,} {g['pct_ce']:>5.1f}% "
          f"{g['mean_rna_seq']:>8.4f} {g['mean_cage']:>8.2f} {g['mean_dnase']:>8.4f} {g['mean_chip_histone']:>8.2f}")

gene_ag.to_csv(f'{OUT_DIR}/R5_only_gene_AG_summary.tsv', sep='\t', index=False)

print(f"\n{'='*70}")
print("Step 7: R4 vs R5-only AlphaGenome comparison")
print(f"{'='*70}")

r4 = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
r4['total_AC'] = r4['case_AC'] + r4['ctrl_AC']
r4_filt = r4[r4['total_AC'] >= 3].copy()

r4_case = r4_filt[r4_filt['enrichment'] == 'case_enriched']
r4_ctrl = r4_filt[r4_filt['enrichment'] == 'ctrl_enriched']

r4_modalities = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

print(f"\n{'Modality':20s} {'R4 CE/CtE':>10s} {'R5 CE/CtE':>10s} {'R4 P-val':>12s} {'R5 P-val':>12s}")
print("-" * 70)

for mod in r4_modalities:
    r4_mod = mod.replace('_effect', '') + '_effect' if not mod.endswith('_effect') else mod

    r4_ce_vals = r4_case[mod].dropna() if mod in r4_case.columns else pd.Series()
    r4_cte_vals = r4_ctrl[mod].dropna() if mod in r4_ctrl.columns else pd.Series()

    r5_ce_vals = case_enr[mod].dropna()
    r5_cte_vals = ctrl_enr[mod].dropna()

    r4_ratio = r4_ce_vals.mean() / r4_cte_vals.mean() if len(r4_cte_vals) > 0 and r4_cte_vals.mean() > 0 else np.nan
    r5_ratio = r5_ce_vals.mean() / r5_cte_vals.mean() if len(r5_cte_vals) > 0 and r5_cte_vals.mean() > 0 else np.nan

    try:
        _, r4_p = stats.mannwhitneyu(r4_ce_vals, r4_cte_vals) if len(r4_ce_vals) > 10 and len(r4_cte_vals) > 10 else (np.nan, np.nan)
    except:
        r4_p = np.nan
    try:
        _, r5_p = stats.mannwhitneyu(r5_ce_vals, r5_cte_vals) if len(r5_ce_vals) > 10 and len(r5_cte_vals) > 10 else (np.nan, np.nan)
    except:
        r5_p = np.nan

    r4_str = f"{r4_ratio:.3f}" if not np.isnan(r4_ratio) else "N/A"
    r5_str = f"{r5_ratio:.3f}" if not np.isnan(r5_ratio) else "N/A"
    r4_p_str = f"{r4_p:.2e}" if not np.isnan(r4_p) else "N/A"
    r5_p_str = f"{r5_p:.2e}" if not np.isnan(r5_p) else "N/A"

    print(f"{mod:20s} {r4_str:>10s} {r5_str:>10s} {r4_p_str:>12s} {r5_p_str:>12s}")

print(f"\n{'='*70}")
print("Step 8: Saving output files")
print(f"{'='*70}")

output_cols = ['chr_num', 'variant_id', 'pos', 'REF', 'ALT',
               'case_AF', 'case_AN', 'case_AC', 'ctrl_AF', 'ctrl_AN', 'ctrl_AC',
               'total_AC', 'cc_ratio', 'log2_cc_ratio', 'enrichment',
               'gene_name', 'gene_id'] + AG_MODALITIES + ['ag_match_source']

out_path = f'{OUT_DIR}/R5_only_variants_with_AG.tsv'
r5v[output_cols].to_csv(out_path, sep='\t', index=False)
print(f"Saved: {out_path} ({len(r5v):,} variants)")

matched_path = f'{OUT_DIR}/R5_only_variants_AG_matched.tsv'
matched[output_cols].to_csv(matched_path, sep='\t', index=False)
print(f"Saved: {matched_path} ({len(matched):,} variants)")

print(f"\n{'='*70}")
print("PROMPT 3 COMPLETE")
print(f"{'='*70}")
print(f"R5-only variants: {len(r5v):,}")
print(f"AG matched: {total_matched:,} ({total_matched/len(r5v)*100:.1f}%)")
print(f"  From R5 AG: {(r5v['ag_match_source'].str.startswith('r5_ag')).sum():,}")
print(f"  From R4 AG: {(r5v['ag_match_source'].str.startswith('r4_ag')).sum():,}")
print(f"Unmatched: {(r5v['ag_match_source']=='unmatched').sum():,}")
print(f"\nFinished: {datetime.now()}")
