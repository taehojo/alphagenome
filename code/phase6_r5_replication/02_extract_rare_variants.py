import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime

def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

OUT_DIR = '<WORK_DIR>/analysis/r5_replication'
R5_PLINK = '<ADSP_R5_PLINK_DIR>'
GENE_LIST = '<ADSP_AI_DIR>/R5/resilience_variants/new_genelist/AD_86genes_list.csv'
R4_DATA = '<WORK_DIR>/data/variant_cc_with_alphgenome.csv'
SAMPLES_FILE = f'{OUT_DIR}/R5_only_samples.tsv'

PROMOTER_REGION = 5000
UTR3_REGION = 1000
MAF_THRESHOLD = 0.01
AC_THRESHOLD = 3

PLINK2 = '<PLINK2_BIN>'
PLINK1 = '<PLINK_BIN>'

os.makedirs(f'{OUT_DIR}/plink_temp', exist_ok=True)

print(f"{'='*70}")
print(f"Prompt 2: R5-only Rare Variant Extraction")
print(f"Started: {datetime.now()}")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("Step 1: Creating gene regions file (5kb upstream, 1kb downstream)")
print(f"{'='*70}")

genes = pd.read_csv(GENE_LIST)
print(f"Loaded {len(genes)} genes from {GENE_LIST}")

regions = []
for _, g in genes.iterrows():
    chrom = g['chr']
    start = max(1, g['start'] - PROMOTER_REGION)
    end = g['end'] + UTR3_REGION
    regions.append({
        'chr': chrom,
        'start': start,
        'end': end,
        'gene_name': g['gene_name'],
        'gene_start': g['start'],
        'gene_end': g['end']
    })

regions_df = pd.DataFrame(regions)

range_file = f'{OUT_DIR}/plink_temp/gene_regions_5kb1kb.txt'
regions_df[['chr', 'start', 'end', 'gene_name']].to_csv(
    range_file, sep='\t', index=False, header=False
)
print(f"Saved: {range_file}")

chr_counts = regions_df['chr'].value_counts().sort_index()
chrs_with_genes = sorted(regions_df['chr'].unique())
print(f"\nChromosomes with genes: {chrs_with_genes}")
print(f"Genes per chromosome:")
for c in chrs_with_genes:
    gene_names = regions_df[regions_df['chr'] == c]['gene_name'].tolist()
    print(f"  chr{c}: {len(gene_names)} genes ({', '.join(gene_names[:5])}{'...' if len(gene_names) > 5 else ''})")

print(f"\n{'='*70}")
print("Step 2: Creating case/control keep files")
print(f"{'='*70}")

samples = pd.read_csv(SAMPLES_FILE, sep='\t')
print(f"R5-only samples: {len(samples):,}")
print(f"  AD: {(samples['Diagnosis']=='AD').sum():,}")
print(f"  CN: {(samples['Diagnosis']=='CN').sum():,}")

case_samples = samples[samples['Diagnosis'] == 'AD']
case_keep = f'{OUT_DIR}/plink_temp/R5_only_case_keep.txt'
pd.DataFrame({'FID': 0, 'IID': case_samples['IID'].values}).to_csv(
    case_keep, sep='\t', index=False, header=False
)
print(f"Case keep file: {case_keep} ({len(case_samples):,} samples)")

ctrl_samples = samples[samples['Diagnosis'] == 'CN']
ctrl_keep = f'{OUT_DIR}/plink_temp/R5_only_ctrl_keep.txt'
pd.DataFrame({'FID': 0, 'IID': ctrl_samples['IID'].values}).to_csv(
    ctrl_keep, sep='\t', index=False, header=False
)
print(f"Control keep file: {ctrl_keep} ({len(ctrl_samples):,} samples)")

all_keep = f'{OUT_DIR}/R5_only_keep.txt'
print(f"All keep file: {all_keep} ({len(samples):,} samples)")

print(f"\n{'='*70}")
print("Step 3: PLINK2 rare variant extraction per chromosome")
print(f"{'='*70}")

extraction_stats = []

for chrom in chrs_with_genes:
    print(f"\n--- Chromosome {chrom} ---")
    t0 = datetime.now()

    bfile = f'{R5_PLINK}/chr{chrom}'
    out_prefix = f'{OUT_DIR}/plink_temp/r5only_rare_chr{chrom}'

    if not os.path.exists(f'{bfile}.bed'):
        print(f"  ERROR: {bfile}.bed not found, skipping")
        continue

    cmd = [
        PLINK2,
        '--bfile', bfile,
        '--keep', all_keep,
        '--extract', 'range', range_file,
        '--max-maf', str(MAF_THRESHOLD),
        '--mac', '1',
        '--make-bed',
        '--out', out_prefix,
        '--threads', '8',
        '--memory', '16000'
    ]

    result = run_cmd(cmd)
    if result.returncode != 0:
        print(f"  PLINK2 extraction ERROR: {result.stderr[-500:]}")
        continue

    bim_file = f'{out_prefix}.bim'
    if not os.path.exists(bim_file):
        print(f"  WARNING: No variants extracted for chr{chrom}")
        extraction_stats.append({'chr': chrom, 'n_variants': 0})
        continue

    n_variants = sum(1 for _ in open(bim_file))
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Extracted: {n_variants:,} rare variants ({elapsed:.0f}s)")

    case_out = f'{OUT_DIR}/plink_temp/r5only_case_freq_chr{chrom}'
    cmd_case = [
        PLINK2,
        '--bfile', out_prefix,
        '--keep', case_keep,
        '--freq', 'counts',
        '--out', case_out,
        '--threads', '4'
    ]
    result_case = run_cmd(cmd_case)
    if result_case.returncode != 0:
        print(f"  Case freq ERROR: {result_case.stderr[-300:]}")

    ctrl_out = f'{OUT_DIR}/plink_temp/r5only_ctrl_freq_chr{chrom}'
    cmd_ctrl = [
        PLINK2,
        '--bfile', out_prefix,
        '--keep', ctrl_keep,
        '--freq', 'counts',
        '--out', ctrl_out,
        '--threads', '4'
    ]
    result_ctrl = run_cmd(cmd_ctrl)
    if result_ctrl.returncode != 0:
        print(f"  Ctrl freq ERROR: {result_ctrl.stderr[-300:]}")

    extraction_stats.append({'chr': chrom, 'n_variants': n_variants})

print(f"\n\n{'='*70}")
print("Extraction Summary")
print(f"{'='*70}")
stats_df = pd.DataFrame(extraction_stats)
total_extracted = stats_df['n_variants'].sum()
print(f"Total rare variants extracted: {total_extracted:,}")
for _, row in stats_df.iterrows():
    print(f"  chr{int(row['chr']):>2d}: {int(row['n_variants']):>8,} variants")

print(f"\n{'='*70}")
print("Step 4: Merging case/control frequency counts")
print(f"{'='*70}")

all_variants = []

for chrom in chrs_with_genes:
    case_freq_file = f'{OUT_DIR}/plink_temp/r5only_case_freq_chr{chrom}.acount'
    ctrl_freq_file = f'{OUT_DIR}/plink_temp/r5only_ctrl_freq_chr{chrom}.acount'

    if not os.path.exists(case_freq_file) or not os.path.exists(ctrl_freq_file):
        print(f"  chr{chrom}: missing frequency files, skipping")
        continue

    case_df = pd.read_csv(case_freq_file, sep='\t')
    case_df = case_df.rename(columns={
        '#CHROM': 'chr_num', 'ID': 'variant_id', 'REF': 'REF', 'ALT': 'ALT',
        'ALT_CTS': 'case_AC', 'OBS_CT': 'case_AN'
    })

    ctrl_df = pd.read_csv(ctrl_freq_file, sep='\t')
    ctrl_df = ctrl_df.rename(columns={
        'ALT_CTS': 'ctrl_AC', 'OBS_CT': 'ctrl_AN'
    })

    merged = case_df[['chr_num', 'variant_id', 'REF', 'ALT', 'case_AC', 'case_AN']].copy()
    merged['ctrl_AC'] = ctrl_df['ctrl_AC'].values
    merged['ctrl_AN'] = ctrl_df['ctrl_AN'].values

    bim_path = f'{OUT_DIR}/plink_temp/r5only_rare_chr{chrom}.bim'
    if os.path.exists(bim_path):
        bim = pd.read_csv(bim_path, sep='\t', header=None,
                          names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'])
        pos_map = dict(zip(bim['SNP'], bim['POS']))
        merged['pos'] = merged['variant_id'].map(pos_map)
    else:
        merged['pos'] = np.nan

    all_variants.append(merged)
    print(f"  chr{chrom}: {len(merged):,} variants")

if not all_variants:
    print("ERROR: No variants extracted! Aborting.")
    import sys
    sys.exit(1)

variants = pd.concat(all_variants, ignore_index=True)
print(f"\nTotal merged variants: {len(variants):,}")

print(f"\n{'='*70}")
print("Step 5: AC >= 3 filter and CC_ratio calculation")
print(f"{'='*70}")

variants['total_AC'] = variants['case_AC'] + variants['ctrl_AC']
print(f"Variants before AC filter: {len(variants):,}")
print(f"  AC=1: {(variants['total_AC']==1).sum():,}")
print(f"  AC=2: {(variants['total_AC']==2).sum():,}")
print(f"  AC>=3: {(variants['total_AC']>=AC_THRESHOLD).sum():,}")

v_filt = variants[variants['total_AC'] >= AC_THRESHOLD].copy()
print(f"\nAfter AC >= {AC_THRESHOLD} filter: {len(v_filt):,} variants")

v_filt['case_AF'] = v_filt['case_AC'] / v_filt['case_AN']
v_filt['ctrl_AF'] = v_filt['ctrl_AC'] / v_filt['ctrl_AN']

v_filt['cc_ratio'] = np.where(
    v_filt['ctrl_AF'] > 0,
    v_filt['case_AF'] / v_filt['ctrl_AF'],
    np.where(v_filt['case_AF'] > 0, np.inf, np.nan)
)

v_filt['enrichment'] = np.where(
    v_filt['cc_ratio'] > 1, 'case_enriched',
    np.where(v_filt['cc_ratio'] < 1, 'ctrl_enriched',
    np.where(v_filt['cc_ratio'] == 1, 'neutral', 'NA'))
)

v_filt['log2_cc_ratio'] = np.where(
    (v_filt['cc_ratio'] > 0) & (v_filt['cc_ratio'] != np.inf),
    np.log2(v_filt['cc_ratio']),
    np.nan
)

print(f"\nCC_ratio distribution:")
case_enr = (v_filt['enrichment'] == 'case_enriched').sum()
ctrl_enr = (v_filt['enrichment'] == 'ctrl_enriched').sum()
neutral = (v_filt['enrichment'] == 'neutral').sum()
print(f"  Case-enriched (CC>1): {case_enr:,} ({case_enr/len(v_filt)*100:.1f}%)")
print(f"  Ctrl-enriched (CC<1): {ctrl_enr:,} ({ctrl_enr/len(v_filt)*100:.1f}%)")
print(f"  Neutral (CC=1):       {neutral:,}")
print(f"  Infinite CC (ctrl_AF=0): {(v_filt['cc_ratio']==np.inf).sum():,}")

print(f"\n{'='*70}")
print("Step 6: Mapping variants to genes")
print(f"{'='*70}")

gene_intervals = {}
for _, g in genes.iterrows():
    chrom = g['chr']
    if chrom not in gene_intervals:
        gene_intervals[chrom] = []
    gene_intervals[chrom].append({
        'start': max(1, g['start'] - PROMOTER_REGION),
        'end': g['end'] + UTR3_REGION,
        'gene_name': g['gene_name'],
        'gene_id': g['gene_id']
    })

def map_to_gene(row):
    chrom = row['chr_num']
    pos = row['pos']
    if pd.isna(pos) or chrom not in gene_intervals:
        return None, None
    for gi in gene_intervals[chrom]:
        if gi['start'] <= pos <= gi['end']:
            return gi['gene_name'], gi['gene_id']
    return None, None

gene_names = []
gene_ids = []
for _, row in v_filt.iterrows():
    gn, gid = map_to_gene(row)
    gene_names.append(gn)
    gene_ids.append(gid)

v_filt['gene_name'] = gene_names
v_filt['gene_id'] = gene_ids

unmapped = v_filt['gene_name'].isna().sum()
print(f"Variants mapped to genes: {len(v_filt) - unmapped:,}")
print(f"Unmapped (should be 0): {unmapped:,}")

if unmapped > 0:
    print("  WARNING: Some variants could not be mapped to genes")
    v_filt = v_filt[v_filt['gene_name'].notna()].copy()
    print(f"  After removing unmapped: {len(v_filt):,}")

gene_summary = v_filt.groupby('gene_name').agg(
    n_variants=('variant_id', 'count'),
    n_case_enriched=('enrichment', lambda x: (x == 'case_enriched').sum()),
    mean_cc=('cc_ratio', lambda x: x[x != np.inf].mean() if (x != np.inf).any() else np.nan),
    median_cc=('cc_ratio', lambda x: x[x != np.inf].median() if (x != np.inf).any() else np.nan)
).reset_index()

gene_summary['pct_case_enriched'] = (gene_summary['n_case_enriched'] / gene_summary['n_variants'] * 100).round(1)
gene_summary = gene_summary.sort_values('n_variants', ascending=False)

print(f"\nGene-level summary ({len(gene_summary)} genes with variants):")
print(f"{'Gene':15s} {'N_var':>7s} {'N_CE':>6s} {'%CE':>6s} {'MeanCC':>8s} {'MedCC':>7s}")
print("-" * 55)
for _, g in gene_summary.head(20).iterrows():
    mean_cc = f"{g['mean_cc']:.3f}" if pd.notna(g['mean_cc']) else "N/A"
    med_cc = f"{g['median_cc']:.3f}" if pd.notna(g['median_cc']) else "N/A"
    print(f"{g['gene_name']:15s} {g['n_variants']:>7,} {g['n_case_enriched']:>6,} {g['pct_case_enriched']:>5.1f}% {mean_cc:>8s} {med_cc:>7s}")

print(f"  ... ({len(gene_summary)} total genes)")

print(f"\n{'='*70}")
print("Step 7: R4 vs R5-only comparison")
print(f"{'='*70}")

r4 = pd.read_csv(R4_DATA)
r4['total_AC'] = r4['case_AC'] + r4['ctrl_AC']
r4_filt = r4[r4['total_AC'] >= AC_THRESHOLD].copy()
r4_unique = r4_filt.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')

r4_gene = r4_unique.groupby('gene_name').agg(
    r4_n_variants=('variant_id', 'count'),
    r4_n_case_enriched=('enrichment', lambda x: (x == 'case_enriched').sum()),
    r4_pct_ce=('enrichment', lambda x: (x == 'case_enriched').sum() / len(x) * 100)
).reset_index()

r5_gene = gene_summary[['gene_name', 'n_variants', 'n_case_enriched', 'pct_case_enriched']].copy()
r5_gene = r5_gene.rename(columns={
    'n_variants': 'r5_n_variants',
    'n_case_enriched': 'r5_n_case_enriched',
    'pct_case_enriched': 'r5_pct_ce'
})

comparison = r4_gene.merge(r5_gene, on='gene_name', how='outer')
comparison = comparison.sort_values('r4_n_variants', ascending=False)

print(f"\nR4 (N=24,595): {len(r4_unique):,} unique variants (AC≥3), {r4_gene['gene_name'].nunique()} genes")
print(f"R5-only (N=14,778): {len(v_filt):,} variants (AC≥3), {r5_gene['gene_name'].nunique()} genes")

r4_ce_pct = (r4_unique['enrichment'] == 'case_enriched').sum() / len(r4_unique) * 100
r5_ce_pct = case_enr / len(v_filt) * 100
print(f"\nOverall case-enriched %:")
print(f"  R4:      {(r4_unique['enrichment']=='case_enriched').sum():,} / {len(r4_unique):,} ({r4_ce_pct:.1f}%)")
print(f"  R5-only: {case_enr:,} / {len(v_filt):,} ({r5_ce_pct:.1f}%)")

print(f"\n{'Gene':15s} {'R4_var':>7s} {'R5_var':>7s} {'R4_%CE':>7s} {'R5_%CE':>7s}")
print("-" * 50)
for _, g in comparison.head(25).iterrows():
    r4_v = f"{int(g['r4_n_variants']):,}" if pd.notna(g['r4_n_variants']) else "-"
    r5_v = f"{int(g['r5_n_variants']):,}" if pd.notna(g['r5_n_variants']) else "-"
    r4_p = f"{g['r4_pct_ce']:.1f}%" if pd.notna(g['r4_pct_ce']) else "-"
    r5_p = f"{g['r5_pct_ce']:.1f}%" if pd.notna(g['r5_pct_ce']) else "-"
    print(f"{g['gene_name']:15s} {r4_v:>7s} {r5_v:>7s} {r4_p:>7s} {r5_p:>7s}")

print(f"\n{'='*70}")
print("Step 8: Saving output files")
print(f"{'='*70}")

output_cols = ['chr_num', 'variant_id', 'pos', 'REF', 'ALT',
               'case_AF', 'case_AN', 'case_AC', 'ctrl_AF', 'ctrl_AN', 'ctrl_AC',
               'total_AC', 'cc_ratio', 'log2_cc_ratio', 'enrichment',
               'gene_name', 'gene_id']
v_out = v_filt[output_cols].copy()
variant_path = f'{OUT_DIR}/R5_only_variants_AC3.tsv'
v_out.to_csv(variant_path, sep='\t', index=False)
print(f"Saved: {variant_path} ({len(v_out):,} variants)")

gene_path = f'{OUT_DIR}/R5_only_gene_summary.tsv'
gene_summary.to_csv(gene_path, sep='\t', index=False)
print(f"Saved: {gene_path} ({len(gene_summary)} genes)")

comp_path = f'{OUT_DIR}/R4_vs_R5_gene_comparison.tsv'
comparison.to_csv(comp_path, sep='\t', index=False)
print(f"Saved: {comp_path}")

print(f"\n{'='*70}")
print("Step 9: Population-specific variant counts")
print(f"{'='*70}")

pop_samples = samples.groupby('Population')
pop_results = {}

for pop, grp in pop_samples:
    n_ad = (grp['Diagnosis'] == 'AD').sum()
    n_cn = (grp['Diagnosis'] == 'CN').sum()
    if n_ad < 10 or n_cn < 10:
        print(f"  {pop}: AD={n_ad}, CN={n_cn} - too few, skipping population-specific analysis")
        continue

    print(f"\n  {pop}: AD={n_ad:,}, CN={n_cn:,}")

    pop_case = grp[grp['Diagnosis'] == 'AD']
    pop_ctrl = grp[grp['Diagnosis'] == 'CN']

    pop_case_keep = f'{OUT_DIR}/plink_temp/R5_{pop}_case_keep.txt'
    pop_ctrl_keep = f'{OUT_DIR}/plink_temp/R5_{pop}_ctrl_keep.txt'
    pop_all_keep = f'{OUT_DIR}/plink_temp/R5_{pop}_all_keep.txt'

    pd.DataFrame({'FID': 0, 'IID': pop_case['IID'].values}).to_csv(
        pop_case_keep, sep='\t', index=False, header=False)
    pd.DataFrame({'FID': 0, 'IID': pop_ctrl['IID'].values}).to_csv(
        pop_ctrl_keep, sep='\t', index=False, header=False)
    pd.DataFrame({'FID': 0, 'IID': grp['IID'].values}).to_csv(
        pop_all_keep, sep='\t', index=False, header=False)

    pop_variant_dfs = []
    for chrom in chrs_with_genes:
        bfile = f'{R5_PLINK}/chr{chrom}'
        if not os.path.exists(f'{bfile}.bed'):
            continue

        pop_out = f'{OUT_DIR}/plink_temp/r5_{pop}_rare_chr{chrom}'
        cmd = [
            PLINK2,
            '--bfile', bfile,
            '--keep', pop_all_keep,
            '--extract', 'range', range_file,
            '--max-maf', str(MAF_THRESHOLD),
            '--mac', '1',
            '--make-bed',
            '--out', pop_out,
            '--threads', '4',
            '--memory', '8000'
        ]
        result = run_cmd(cmd)
        if result.returncode != 0:
            continue

        if not os.path.exists(f'{pop_out}.bim'):
            continue

        pop_case_out = f'{OUT_DIR}/plink_temp/r5_{pop}_case_freq_chr{chrom}'
        run_cmd([
            PLINK2, '--bfile', pop_out, '--keep', pop_case_keep,
            '--freq', 'counts', '--out', pop_case_out, '--threads', '2'
        ])

        pop_ctrl_out = f'{OUT_DIR}/plink_temp/r5_{pop}_ctrl_freq_chr{chrom}'
        run_cmd([
            PLINK2, '--bfile', pop_out, '--keep', pop_ctrl_keep,
            '--freq', 'counts', '--out', pop_ctrl_out, '--threads', '2'
        ])

        cf = f'{pop_case_out}.acount'
        ctf = f'{pop_ctrl_out}.acount'
        if os.path.exists(cf) and os.path.exists(ctf):
            cdf = pd.read_csv(cf, sep='\t')
            ctdf = pd.read_csv(ctf, sep='\t')

            bim = pd.read_csv(f'{pop_out}.bim', sep='\t', header=None,
                              names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'])
            pos_map = dict(zip(bim['SNP'], bim['POS']))

            mdf = pd.DataFrame({
                'chr_num': cdf['#CHROM'],
                'variant_id': cdf['ID'],
                'pos': cdf['ID'].map(pos_map),
                'REF': cdf['REF'],
                'ALT': cdf['ALT'],
                'case_AC': cdf['ALT_CTS'],
                'case_AN': cdf['OBS_CT'],
                'ctrl_AC': ctdf['ALT_CTS'],
                'ctrl_AN': ctdf['OBS_CT']
            })
            pop_variant_dfs.append(mdf)

    if pop_variant_dfs:
        pop_vars = pd.concat(pop_variant_dfs, ignore_index=True)
        pop_vars['total_AC'] = pop_vars['case_AC'] + pop_vars['ctrl_AC']
        pop_vars_filt = pop_vars[pop_vars['total_AC'] >= AC_THRESHOLD].copy()

        gn_list = []
        for _, row in pop_vars_filt.iterrows():
            gn, _ = map_to_gene(row)
            gn_list.append(gn)
        pop_vars_filt['gene_name'] = gn_list
        pop_vars_filt = pop_vars_filt[pop_vars_filt['gene_name'].notna()]

        pop_vars_filt['case_AF'] = pop_vars_filt['case_AC'] / pop_vars_filt['case_AN']
        pop_vars_filt['ctrl_AF'] = pop_vars_filt['ctrl_AC'] / pop_vars_filt['ctrl_AN']
        pop_vars_filt['cc_ratio'] = np.where(
            pop_vars_filt['ctrl_AF'] > 0,
            pop_vars_filt['case_AF'] / pop_vars_filt['ctrl_AF'],
            np.where(pop_vars_filt['case_AF'] > 0, np.inf, np.nan)
        )
        pop_vars_filt['enrichment'] = np.where(
            pop_vars_filt['cc_ratio'] > 1, 'case_enriched',
            np.where(pop_vars_filt['cc_ratio'] < 1, 'ctrl_enriched', 'neutral'))

        ce_n = (pop_vars_filt['enrichment'] == 'case_enriched').sum()
        n_genes = pop_vars_filt['gene_name'].nunique()
        print(f"    Variants (AC≥3): {len(pop_vars_filt):,}, Genes: {n_genes}, Case-enriched: {ce_n} ({ce_n/len(pop_vars_filt)*100:.1f}%)")

        pop_results[pop] = {
            'n_samples': len(grp),
            'n_ad': n_ad,
            'n_cn': n_cn,
            'n_variants': len(pop_vars_filt),
            'n_genes': n_genes,
            'n_case_enriched': ce_n,
            'pct_case_enriched': round(ce_n/len(pop_vars_filt)*100, 1) if len(pop_vars_filt) > 0 else 0
        }

        pop_var_path = f'{OUT_DIR}/R5_only_{pop}_variants_AC3.tsv'
        pop_vars_filt.to_csv(pop_var_path, sep='\t', index=False)
        print(f"    Saved: {pop_var_path}")

if pop_results:
    print(f"\n{'Population':12s} {'N':>7s} {'AD':>6s} {'CN':>7s} {'Variants':>9s} {'Genes':>6s} {'%CE':>6s}")
    print("-" * 60)
    for pop in ['NHW', 'AA', 'Hispanic', 'Asian', 'Other']:
        if pop in pop_results:
            p = pop_results[pop]
            print(f"{pop:12s} {p['n_samples']:>7,} {p['n_ad']:>6,} {p['n_cn']:>7,} {p['n_variants']:>9,} {p['n_genes']:>6} {p['pct_case_enriched']:>5.1f}%")

pop_summary_path = f'{OUT_DIR}/R5_only_population_variant_summary.tsv'
if pop_results:
    pop_df = pd.DataFrame(pop_results).T
    pop_df.index.name = 'Population'
    pop_df.to_csv(pop_summary_path, sep='\t')
    print(f"\nSaved: {pop_summary_path}")

print(f"\n{'='*70}")
print("PROMPT 2 COMPLETE")
print(f"{'='*70}")
print(f"R5-only cohort: {len(samples):,} samples (AD {(samples['Diagnosis']=='AD').sum():,}, CN {(samples['Diagnosis']=='CN').sum():,})")
print(f"Total rare variants (AC≥3): {len(v_filt):,}")
print(f"Genes with variants: {len(gene_summary)}")
print(f"Case-enriched: {case_enr:,} ({case_enr/len(v_filt)*100:.1f}%)")
print(f"Ctrl-enriched: {ctrl_enr:,} ({ctrl_enr/len(v_filt)*100:.1f}%)")
print(f"\nR4 comparison:")
print(f"  R4: {len(r4_unique):,} variants, R5-only: {len(v_filt):,} variants")
print(f"  R4 CE%: {r4_ce_pct:.1f}%, R5-only CE%: {r5_ce_pct:.1f}%")
print(f"\nFinished: {datetime.now()}")
