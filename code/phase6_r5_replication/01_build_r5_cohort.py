import pandas as pd
import numpy as np
import os

OUT_DIR = '<WORK_DIR>/analysis/r5_replication'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("Step 1: Loading R4 matched samples")
print("=" * 70)

r4 = pd.read_csv('<ADSP_AI_DIR>/LD-rarevariant-5th-all/matched_samples.tsv', sep='\t')
r4_subjids = set(r4['SUBJID'].values)
print(f"R4 matched samples: {len(r4_subjids):,}")
print(f"  AD: {(r4['PrevAD']==1).sum():,}, CN: {(r4['PrevAD']==0).sum():,}")

print(f"\n{'=' * 70}")
print("Step 2: Loading R5 PLINK .fam")
print("=" * 70)

fam_cols = ['FID', 'IID', 'Father', 'Mother', 'Sex', 'Phenotype']
r5_fam = pd.read_csv('<ADSP_R5_PLINK_DIR>/chr1.fam',
                      sep=r'\s+', header=None, names=fam_cols)
print(f"R5 PLINK samples: {len(r5_fam):,}")

r5_fam['SUBJID'] = r5_fam['IID'].apply(lambda x: '-'.join(str(x).split('-')[:3]))
r5_iid_to_subjid = dict(zip(r5_fam['IID'], r5_fam['SUBJID']))
r5_subjids = set(r5_fam['SUBJID'].values)
print(f"R5 unique SUBJIDs: {len(r5_subjids):,}")

print(f"\n{'=' * 70}")
print("Step 3: Loading R5 phenotype")
print("=" * 70)

pheno = pd.read_csv('<ADSP_AI_DIR>/R4-2025/ADSPCaseControlPhenotypes_DS_2024.11.22_ALL.csv')
print(f"Phenotype records: {len(pheno):,}")
print(f"Unique SUBJIDs: {pheno['SUBJID'].nunique():,}")

print(f"\n{'=' * 70}")
print("Step 4: Matching R5 PLINK to phenotype")
print("=" * 70)

pheno_matched = pheno[pheno['SUBJID'].isin(r5_subjids)].copy()

pheno_matched = pheno_matched.drop_duplicates('SUBJID', keep='first')
print(f"R5 PLINK samples matched to phenotype: {len(pheno_matched):,}")
print(f"Unmatched: {len(r5_subjids) - len(pheno_matched):,}")

print(f"\n{'=' * 70}")
print("Step 5: Removing R4 samples")
print("=" * 70)

r5only_pheno = pheno_matched[~pheno_matched['SUBJID'].isin(r4_subjids)].copy()
print(f"R5-only matched to phenotype: {len(r5only_pheno):,}")
print(f"  (R5 matched {len(pheno_matched):,} - R4 {len(pheno_matched[pheno_matched['SUBJID'].isin(r4_subjids)]):,} = {len(r5only_pheno):,})")

r4_in_r5 = r4_subjids.intersection(r5_subjids)
print(f"\n  R4 SUBJIDs found in R5 PLINK: {len(r4_in_r5):,} / {len(r4_subjids):,}")
r4_not_in_r5 = r4_subjids - r5_subjids
if r4_not_in_r5:
    print(f"  R4 SUBJIDs NOT in R5 PLINK: {len(r4_not_in_r5):,}")

print(f"\n{'=' * 70}")
print("Step 6: Population classification")
print("=" * 70)

def classify_population(row):
    race = row.get('Race', np.nan)
    eth = row.get('Ethnicity', np.nan)
    if eth == 1:
        return 'Hispanic'
    elif race == 5 and eth == 0:
        return 'NHW'
    elif race == 4 and eth == 0:
        return 'AA'
    elif race == 2:
        return 'Asian'
    else:
        return 'Other'

r5only_pheno['Population'] = r5only_pheno.apply(classify_population, axis=1)

r5only_pheno['AD_status'] = r5only_pheno['PrevAD'].map({1: 'AD', 0: 'CN'})
r5only_pheno.loc[r5only_pheno['AD_status'].isna(), 'AD_status'] = 'NA'

r5only_adcn = r5only_pheno[r5only_pheno['AD_status'].isin(['AD', 'CN'])].copy()
print(f"R5-only with AD/CN status: {len(r5only_adcn):,}")
print(f"  Excluded (AD status NA): {len(r5only_pheno) - len(r5only_adcn):,}")

print(f"\n{'=' * 70}")
print("Step 7: R5-only demographics")
print("=" * 70)

print(f"\nOverall: {len(r5only_adcn):,}")
print(f"  AD: {(r5only_adcn['AD_status']=='AD').sum():,}")
print(f"  CN: {(r5only_adcn['AD_status']=='CN').sum():,}")

print(f"\nBy Population:")
summary_rows = []
for pop in ['NHW', 'AA', 'Hispanic', 'Asian', 'Other']:
    subset = r5only_adcn[r5only_adcn['Population'] == pop]
    ad_n = (subset['AD_status'] == 'AD').sum()
    cn_n = (subset['AD_status'] == 'CN').sum()
    total = len(subset)
    print(f"  {pop:12s}: Total={total:>6,}  AD={ad_n:>5,}  CN={cn_n:>5,}")
    summary_rows.append({
        'Population': pop, 'Total': total, 'AD': ad_n, 'CN': cn_n
    })

print(f"\n{'=' * 70}")
print("R4 vs R5-only comparison")
print("=" * 70)

r4_pops = {
    'NHW': {'Total': 8633, 'AD': 3245, 'CN': 5388},
    'AA': {'Total': 4620, 'AD': 1114, 'CN': 3506},
    'Hispanic': {'Total': 8743, 'AD': 1898, 'CN': 6845},
    'Asian': {'Total': 2599, 'AD': 39, 'CN': 2560},
}

print(f"{'Population':12s} {'R4 Total':>10s} {'R5-only Total':>14s} {'R4 AD':>7s} {'R5 AD':>7s} {'R4 CN':>7s} {'R5 CN':>7s}")
print("-" * 72)
for pop in ['NHW', 'AA', 'Hispanic', 'Asian']:
    r4p = r4_pops[pop]
    r5_sub = r5only_adcn[r5only_adcn['Population'] == pop]
    r5_ad = (r5_sub['AD_status'] == 'AD').sum()
    r5_cn = (r5_sub['AD_status'] == 'CN').sum()
    print(f"{pop:12s} {r4p['Total']:>10,} {len(r5_sub):>14,} {r4p['AD']:>7,} {r5_ad:>7,} {r4p['CN']:>7,} {r5_cn:>7,}")

asian_r5 = r5only_adcn[(r5only_adcn['Population'] == 'Asian') & (r5only_adcn['AD_status'] == 'AD')]
print(f"\nAsian AD in R5-only: {len(asian_r5):,}")
if len(asian_r5) < 50:
    print("  WARNING: Asian AD count may be insufficient for independent analysis")

print(f"\n{'=' * 70}")
print("Step 8: Creating output files")
print("=" * 70)

subjid_to_iid = {}
for _, row in r5_fam.iterrows():
    subjid = row['SUBJID']
    if subjid not in subjid_to_iid:
        subjid_to_iid[subjid] = row['IID']

r5only_adcn = r5only_adcn.copy()
r5only_adcn['IID'] = r5only_adcn['SUBJID'].map(subjid_to_iid)

r5only_adcn['Age_clean'] = pd.to_numeric(r5only_adcn['Age'].replace('90+', '90'), errors='coerce')

r5only_adcn['Sex_label'] = r5only_adcn['Sex'].map({0: 'Female', 1: 'Male'})

out_samples = r5only_adcn[['SUBJID', 'IID', 'AD_status', 'Population', 'Sex_label',
                            'Age', 'APOE_reported', 'Cohort']].copy()
out_samples.columns = ['SUBJID', 'IID', 'Diagnosis', 'Population', 'Sex', 'Age', 'APOE', 'Cohort']
samples_path = f'{OUT_DIR}/R5_only_samples.tsv'
out_samples.to_csv(samples_path, sep='\t', index=False)
print(f"Saved: {samples_path} ({len(out_samples):,} samples)")

keep_path = f'{OUT_DIR}/R5_only_keep.txt'
keep_df = pd.DataFrame({'FID': 0, 'IID': r5only_adcn['IID'].values})
keep_df.to_csv(keep_path, sep='\t', index=False, header=False)
print(f"Saved: {keep_path} ({len(keep_df):,} samples)")

summary_df = pd.DataFrame(summary_rows)
summary_path = f'{OUT_DIR}/R5_only_demographics_summary.tsv'
summary_df.to_csv(summary_path, sep='\t', index=False)
print(f"Saved: {summary_path}")

print(f"\n{'=' * 70}")
print("Additional demographics for report")
print("=" * 70)

ad = r5only_adcn[r5only_adcn['AD_status'] == 'AD']
cn = r5only_adcn[r5only_adcn['AD_status'] == 'CN']

for label, grp in [('AD', ad), ('CN', cn), ('Total', r5only_adcn)]:
    age_num = pd.to_numeric(grp['Age'].replace('90+', '90'), errors='coerce')
    valid = age_num.dropna()
    n90 = (grp['Age'] == '90+').sum()
    print(f"  {label} Age: {valid.mean():.1f} ± {valid.std():.1f} (n_90+={n90}, missing={age_num.isna().sum()})")

for label, grp in [('AD', ad), ('CN', cn)]:
    f_n = (grp['Sex'] == 0).sum()
    print(f"  {label} Female: {f_n} ({f_n/len(grp)*100:.1f}%)")

for label, grp in [('AD', ad), ('CN', cn)]:
    apoe = grp['APOE_reported'].dropna()
    e4 = apoe.isin([14, 24, 34, 44]).sum()
    print(f"  {label} APOE e4+: {e4}/{len(apoe)} ({e4/len(apoe)*100:.1f}%)" if len(apoe) > 0 else f"  {label} APOE: no data")

print(f"\nPrompt 1 complete!")
