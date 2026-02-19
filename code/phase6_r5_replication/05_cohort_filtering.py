import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '<WORK_DIR>/analysis/r5_replication'
REPLICATION_POPS = ['NHW', 'Hispanic', 'AA']

print("=" * 70)
print("Phase A: Replication Cohort Construction (NHW + Hispanic + AA)")
print("Started: {}".format(datetime.now()))
print("=" * 70)

print("\n--- A-1. Sample Subset ---")
samples_all = pd.read_csv('{}/R5_only_samples.tsv'.format(OUT), sep='\t')
samples = samples_all[samples_all['Population'].isin(REPLICATION_POPS)].copy()

print("R5-only total: {:,}".format(len(samples_all)))
print("Replication cohort (NHW+Hisp+AA): {:,}".format(len(samples)))
print("  Excluded: Asian {:,}, Other {:,}".format(
    len(samples_all[samples_all['Population'] == 'Asian']),
    len(samples_all[samples_all['Population'] == 'Other'])))

print("\nReplication cohort demographics:")
print("{:12s} {:>6s} {:>6s} {:>6s} {:>8s}".format('Population', 'AD', 'CN', 'Total', 'Ratio'))
print("-" * 45)
total_ad = 0
total_cn = 0
pop_stats = []
for pop in REPLICATION_POPS:
    ps = samples[samples['Population'] == pop]
    ad = len(ps[ps['Diagnosis'] == 'AD'])
    cn = len(ps[ps['Diagnosis'] == 'CN'])
    ratio = cn / ad if ad > 0 else 0
    total_ad += ad
    total_cn += cn
    print("{:12s} {:>6d} {:>6d} {:>6d} {:>8.1f}".format(pop, ad, cn, len(ps), ratio))
    pop_stats.append({'Population': pop, 'R5_AD': ad, 'R5_CN': cn, 'R5_Total': len(ps),
                      'R5_ratio': ratio})

print("{:12s} {:>6d} {:>6d} {:>6d} {:>8.1f}".format(
    'TOTAL', total_ad, total_cn, len(samples), total_cn / total_ad))

r4_demo = {'NHW': {'AD': 3245}, 'Hispanic': {'AD': 1898}, 'AA': {'AD': 1114}, 'Asian': {'AD': 39}}

r4_full = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')

print("\n--- R4 vs R5 Replication Cohort ---")
demo_rows = []
for pop in REPLICATION_POPS:
    r4_ad = r4_demo.get(pop, {}).get('AD', 0)
    r5_ad = len(samples[(samples['Population'] == pop) & (samples['Diagnosis'] == 'AD')])
    r5_cn = len(samples[(samples['Population'] == pop) & (samples['Diagnosis'] == 'CN')])
    demo_rows.append({
        'Population': pop,
        'R4_AD': r4_ad,
        'R5_AD': r5_ad,
        'R5_CN': r5_cn,
        'R5_Total': r5_ad + r5_cn,
        'R5_ratio': r5_cn / r5_ad if r5_ad > 0 else 0
    })

demo_rows.append({
    'Population': 'TOTAL',
    'R4_AD': sum(r4_demo[p]['AD'] for p in REPLICATION_POPS),
    'R5_AD': total_ad,
    'R5_CN': total_cn,
    'R5_Total': total_ad + total_cn,
    'R5_ratio': total_cn / total_ad
})

demo_df = pd.DataFrame(demo_rows)
print(demo_df.to_string(index=False))

print("\nAdditional demographics:")
for pop in REPLICATION_POPS + ['ALL']:
    if pop == 'ALL':
        ps = samples
        label = 'ALL'
    else:
        ps = samples[samples['Population'] == pop]
        label = pop
    ad = ps[ps['Diagnosis'] == 'AD']
    cn = ps[ps['Diagnosis'] == 'CN']
    print("  {} (AD={}, CN={}):".format(label, len(ad), len(cn)))
    if 'Age' in ps.columns:
        ad_age = pd.to_numeric(ad['Age'], errors='coerce')
        cn_age = pd.to_numeric(cn['Age'], errors='coerce')
        if ad_age.notna().sum() > 0:
            print("    Age: AD={:.1f}+/-{:.1f}, CN={:.1f}+/-{:.1f}".format(
                ad_age.mean(), ad_age.std(), cn_age.mean(), cn_age.std()))
        else:
            print("    Age: not available")
    if 'Sex' in ps.columns and ps['Sex'].notna().sum() > 0:
        ad_f = (ad['Sex'] == 'F').sum() + (ad['Sex'] == 2).sum()
        cn_f = (cn['Sex'] == 'F').sum() + (cn['Sex'] == 2).sum()
        print("    Female: AD={}/{} ({:.1f}%), CN={}/{} ({:.1f}%)".format(
            ad_f, len(ad), ad_f / len(ad) * 100 if len(ad) > 0 else 0,
            cn_f, len(cn), cn_f / len(cn) * 100 if len(cn) > 0 else 0))
    if 'APOE' in ps.columns:
        ad_apoe = pd.to_numeric(ad['APOE'], errors='coerce')
        cn_apoe = pd.to_numeric(cn['APOE'], errors='coerce')
        ad_e4 = ad_apoe.isin([34, 44, 24]).sum()
        cn_e4 = cn_apoe.isin([34, 44, 24]).sum()
        print("    APOE e4+: AD={}/{} ({:.1f}%), CN={}/{} ({:.1f}%)".format(
            ad_e4, len(ad), ad_e4 / len(ad) * 100 if len(ad) > 0 else 0,
            cn_e4, len(cn), cn_e4 / len(cn) * 100 if len(cn) > 0 else 0))

demo_df.to_csv('{}/Phase_A_replication_cohort_demographics.tsv'.format(OUT),
               sep='\t', index=False)


print("\n\n--- A-2. Variant Filtering ---")

waterfall = []

all_pop_variants = []
for pop in REPLICATION_POPS:
    vf = '{}/R5_only_{}_variants_AC3.tsv'.format(OUT, pop)
    if not os.path.exists(vf):
        print("  WARNING: {} not found".format(vf))
        continue
    vdf = pd.read_csv(vf, sep='\t')
    vdf['source_pop'] = pop
    all_pop_variants.append(vdf)
    print("  {}: {:,} variants".format(pop, len(vdf)))

combined = pd.concat(all_pop_variants, ignore_index=True)
waterfall.append({'Step': '0_raw_pop_combined', 'N_variants': len(combined),
                  'Description': 'NHW + Hispanic + AA pop-specific files concatenated'})
print("\nCombined (with duplicates): {:,}".format(len(combined)))

combined['variant_key'] = combined.apply(
    lambda r: '{}_{}_{}_{}'.format(int(r['chr_num']), int(r['pos']), r['REF'], r['ALT']), axis=1)

combined['total_AC'] = combined['case_AC'] + combined['ctrl_AC']
dedup = combined.sort_values('total_AC', ascending=False).drop_duplicates('variant_key', keep='first')
waterfall.append({'Step': '1_pop_maf_dedup', 'N_variants': len(dedup),
                  'Description': 'Pop-specific MAF<1% + AC>=3, deduplicated by variant_key'})
print("After dedup: {:,}".format(len(dedup)))

dedup['cc_ratio_recalc'] = dedup['case_AF'] / dedup['ctrl_AF']
dedup['cc_ratio_recalc'] = dedup['cc_ratio_recalc'].replace([np.inf, -np.inf], np.nan)

finite = dedup[dedup['cc_ratio_recalc'].notna() & np.isfinite(dedup['cc_ratio_recalc'])].copy()
waterfall.append({'Step': '2_finite_cc_ratio', 'N_variants': len(finite),
                  'Description': 'Finite CC_ratio only'})
print("Finite CC_ratio: {:,}".format(len(finite)))

finite['enrichment'] = finite['cc_ratio_recalc'].apply(
    lambda x: 'case_enriched' if x > 1 else ('control_enriched' if x < 1 else 'neutral'))
finite.loc[(finite['case_AC'] > 0) & (finite['ctrl_AC'] == 0), 'enrichment'] = 'case_only'
finite.loc[(finite['ctrl_AC'] > 0) & (finite['case_AC'] == 0), 'enrichment'] = 'control_only'
finite['is_case_enriched'] = finite['enrichment'].isin(['case_enriched', 'case_only'])


print("\n\n--- A-3. Filtering Results ---")

print("\nFiltering Waterfall:")
wf_df = pd.DataFrame(waterfall)
for _, row in wf_df.iterrows():
    print("  {}: {:,} - {}".format(row['Step'], row['N_variants'], row['Description']))
wf_df.to_csv('{}/Phase_A_filtering_waterfall.tsv'.format(OUT), sep='\t', index=False)

ce_total = finite['is_case_enriched'].mean() * 100
print("\nCase-enrichment: {:,}/{:,} ({:.1f}%)".format(
    finite['is_case_enriched'].sum(), len(finite), ce_total))

print("\nPer-population CE% (variants sourced from each pop):")
for pop in REPLICATION_POPS:
    pv = finite[finite['source_pop'] == pop]
    if len(pv) > 0:
        ce = pv['is_case_enriched'].mean() * 100
        print("  {}: {:,} variants, CE% = {:.1f}%".format(pop, len(pv), ce))

r4_full['total_AC'] = r4_full['case_AC'] + r4_full['ctrl_AC']
r4_dedup = r4_full[r4_full['total_AC'] >= 3].sort_values('total_AC', ascending=False) \
                  .drop_duplicates('variant_id', keep='first')
r4_ce = r4_dedup['enrichment'].isin(['case_enriched', 'case_only']).mean() * 100

r4_no_asian = r4_full[r4_full['population'] != 'Asian'].copy()
r4_no_asian['total_AC'] = r4_no_asian['case_AC'] + r4_no_asian['ctrl_AC']
r4_no_asian_dedup = r4_no_asian[r4_no_asian['total_AC'] >= 3] \
    .sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')
r4_noasian_ce = r4_no_asian_dedup['enrichment'].isin(['case_enriched', 'case_only']).mean() * 100

print("\nComparison with R4:")
print("  R4 all populations: {:,} variants, CE% = {:.1f}%".format(len(r4_dedup), r4_ce))
print("  R4 (NHW+Hisp+AA):  {:,} variants, CE% = {:.1f}%".format(
    len(r4_no_asian_dedup), r4_noasian_ce))
print("  R5 replication:     {:,} variants, CE% = {:.1f}%".format(len(finite), ce_total))
print("  R5/R4 variant ratio: {:.1f}x".format(len(finite) / len(r4_dedup)))

print("\nGene-level summary (top 20 by R5 variant count):")
gene_r5 = finite.groupby('gene_name').agg(
    R5_n=('variant_key', 'count'),
    R5_ce_pct=('is_case_enriched', 'mean')
).reset_index()
gene_r5['R5_ce_pct'] = gene_r5['R5_ce_pct'] * 100

gene_r4 = r4_dedup.groupby('gene_name').agg(
    R4_n=('variant_id', 'count'),
    R4_ce_pct=('enrichment', lambda x: x.isin(['case_enriched', 'case_only']).mean() * 100)
).reset_index()

gene_merged = gene_r5.merge(gene_r4, on='gene_name', how='left')
gene_merged = gene_merged.sort_values('R5_n', ascending=False)

print("{:12s} {:>6s} {:>8s} {:>6s} {:>8s}".format('Gene', 'R5_n', 'R5_CE%', 'R4_n', 'R4_CE%'))
print("-" * 45)
for _, row in gene_merged.head(20).iterrows():
    print("{:12s} {:>6d} {:>8.1f} {:>6.0f} {:>8.1f}".format(
        row['gene_name'], row['R5_n'], row['R5_ce_pct'],
        row['R4_n'] if pd.notna(row['R4_n']) else 0,
        row['R4_ce_pct'] if pd.notna(row['R4_ce_pct']) else 0))

gene_merged.to_csv('{}/Phase_A_gene_summary.tsv'.format(OUT), sep='\t', index=False)

finite.to_csv('{}/Phase_A_replication_variants_filtered.tsv'.format(OUT), sep='\t', index=False)
print("\nSaved: Phase_A_replication_variants_filtered.tsv ({:,} variants)".format(len(finite)))

print("\n" + "=" * 70)
print("PHASE A SUMMARY")
print("=" * 70)
print("\nReplication cohort: {:,} (AD {:,}, CN {:,})".format(
    len(samples), total_ad, total_cn))
print("  NHW: AD={}, CN={}".format(
    len(samples[(samples['Population'] == 'NHW') & (samples['Diagnosis'] == 'AD')]),
    len(samples[(samples['Population'] == 'NHW') & (samples['Diagnosis'] == 'CN')])))
print("  Hispanic: AD={}, CN={}".format(
    len(samples[(samples['Population'] == 'Hispanic') & (samples['Diagnosis'] == 'AD')]),
    len(samples[(samples['Population'] == 'Hispanic') & (samples['Diagnosis'] == 'CN')])))
print("  AA: AD={}, CN={}".format(
    len(samples[(samples['Population'] == 'AA') & (samples['Diagnosis'] == 'AD')]),
    len(samples[(samples['Population'] == 'AA') & (samples['Diagnosis'] == 'CN')])))
print("  Case/Control ratio: 1:{:.1f}".format(total_cn / total_ad))
print("\nFiltered variants: {:,} (R4: {:,}, ratio {:.1f}x)".format(
    len(finite), len(r4_dedup), len(finite) / len(r4_dedup)))
print("CE%: {:.1f}% (R4 all: {:.1f}%, R4 NHW+Hisp+AA: {:.1f}%)".format(
    ce_total, r4_ce, r4_noasian_ce))
print("Genes: {}".format(len(gene_r5)))
print("\nFinished: {}".format(datetime.now()))
