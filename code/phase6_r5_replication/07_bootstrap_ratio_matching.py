import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '<WORK_DIR>/analysis/r5_replication'
TEMP = '{}/plink_temp'.format(OUT)
AG_MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
AG_LABELS = {'rna_seq_effect': 'RNA-seq', 'cage_effect': 'CAGE',
             'dnase_effect': 'DNase', 'chip_histone_effect': 'ChIP-histone'}
REPLICATION_POPS = ['NHW', 'Hispanic', 'AA']
TARGET_RATIO = 2.9
N_BOOTSTRAP = 100

R4_REF = {
    'RNA-seq': {'IR': 1.086}, 'CAGE': {'IR': 1.056},
    'DNase': {'IR': 1.028}, 'ChIP-histone': {'IR': 1.062}
}


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
    return {'IR': ir, 'OR': odds_ratio, 'fisher_p': fisher_p,
            'n_high': len(high), 'n_low': len(low),
            'high_ce_pct': high_ce, 'low_ce_pct': low_ce}


print("=" * 70)
print("Phase C: Bootstrap 1:2.9 Matching (NHW + Hispanic + AA)")
print("Started: {}".format(datetime.now()))
print("=" * 70)

samples_all = pd.read_csv('{}/R5_only_samples.tsv'.format(OUT), sep='\t')
samples = samples_all[samples_all['Population'].isin(REPLICATION_POPS)].copy()
shared_df = pd.read_csv('{}/Phase_B_shared_variants.tsv'.format(OUT), sep='\t')

print("Replication samples: {:,} (AD {:,}, CN {:,})".format(
    len(samples),
    len(samples[samples['Diagnosis'] == 'AD']),
    len(samples[samples['Diagnosis'] == 'CN'])))

case_iids = set(samples[samples['Diagnosis'] == 'AD']['IID'].values)
ctrl_iids = set(samples[samples['Diagnosis'] == 'CN']['IID'].values)

pos_to_vkey = {}
vk_to_ag = {}
vk_to_gene = {}
for _, row in shared_df.iterrows():
    vk = row['variant_key']
    chrom = int(row['chr_num'])
    pos = int(row['pos'])
    pos_to_vkey[(chrom, pos)] = vk
    vk_to_ag[vk] = {mod: row[mod] for mod in AG_MODALITIES if mod in row.index and pd.notna(row[mod])}
    if 'gene_name' in row.index and pd.notna(row['gene_name']):
        vk_to_gene[vk] = row['gene_name']

print("\n--- Reading RAW files ---")
raw_files = {}
for chrom in range(1, 23):
    for suffix in ['_v4', '_v3', '_v2', '']:
        raw = '{}/phaseC_chr{}{}.raw'.format(TEMP, chrom, suffix)
        if os.path.exists(raw) and os.path.getsize(raw) > 1000:
            raw_files[chrom] = raw
            break

print("Found {} chromosome RAW files".format(len(raw_files)))

all_iids = None
variant_data = {}

for chrom, raw_path in sorted(raw_files.items()):
    try:
        raw_df = pd.read_csv(raw_path, sep='\t')
        if all_iids is None:
            all_iids = raw_df['IID'].values

        snp_cols = [c for c in raw_df.columns
                    if c not in ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']]

        mapped = 0
        for col in snp_cols:
            col_no_allele = col.rsplit('_', 1)[0]
            pos = None
            if ':' in col_no_allele:
                try:
                    pos = int(col_no_allele.split(':')[1])
                except:
                    pass
            else:
                parts = col_no_allele.split('_')
                if len(parts) >= 2:
                    try:
                        pos = int(parts[1])
                    except:
                        pass
            if pos is None:
                continue
            vk = pos_to_vkey.get((chrom, pos))
            if vk is None:
                continue
            variant_data[vk] = raw_df[col].values
            mapped += 1

        print("  chr{}: {} mapped".format(chrom, mapped))
    except Exception as e:
        print("  chr{}: ERROR - {}".format(chrom, str(e)[:200]))

print("\nTotal variants: {:,}".format(len(variant_data)))
print("Total samples: {:,}".format(len(all_iids)))

sample_idx = {iid: i for i, iid in enumerate(all_iids)}
rep_iids = set(samples['IID'].values)
case_indices = np.array([sample_idx[iid] for iid in all_iids if iid in case_iids])
ctrl_iid_list = [iid for iid in all_iids if iid in ctrl_iids]
full_ctrl_indices = np.array([sample_idx[iid] for iid in ctrl_iid_list])

print("Cases in genotype data: {:,}".format(len(case_indices)))
print("Controls in genotype data: {:,}".format(len(full_ctrl_indices)))

variant_keys = sorted(variant_data.keys())
geno_matrix = np.column_stack([variant_data[vk] for vk in variant_keys])
print("Genotype matrix: {} x {}".format(*geno_matrix.shape))
del variant_data

pop_ctrl_map = {}
for pop in REPLICATION_POPS:
    pop_cases = samples[(samples['Population'] == pop) & (samples['Diagnosis'] == 'AD')]
    pop_ctrls = samples[(samples['Population'] == pop) & (samples['Diagnosis'] == 'CN')]
    pop_ctrl_iids = [iid for iid in pop_ctrls['IID'].values if iid in sample_idx]
    n_case = len(pop_cases)
    n_target = min(int(n_case * TARGET_RATIO), len(pop_ctrl_iids))
    pop_ctrl_map[pop] = {
        'iids': np.array(pop_ctrl_iids),
        'n_case': n_case,
        'n_target': n_target,
        'n_available': len(pop_ctrl_iids)
    }

pop_case_map = {}
for pop in REPLICATION_POPS:
    pop_cases = samples[(samples['Population'] == pop) & (samples['Diagnosis'] == 'AD')]
    pop_case_iids = [iid for iid in pop_cases['IID'].values if iid in sample_idx]
    pop_case_map[pop] = np.array([sample_idx[iid] for iid in pop_case_iids])

print("\n--- C-1. Downsampling Design ---")
print("{:12s} {:>6s} {:>10s} {:>10s} {:>8s} {:>8s}".format(
    'Population', 'AD', 'CN_avail', 'CN_target', 'Ratio', 'OK?'))
print("-" * 60)
total_case = 0
total_target = 0
for pop in REPLICATION_POPS:
    info = pop_ctrl_map[pop]
    ok = 'YES' if info['n_target'] == int(info['n_case'] * TARGET_RATIO) else 'PARTIAL'
    ratio = info['n_target'] / info['n_case'] if info['n_case'] > 0 else 0
    total_case += info['n_case']
    total_target += info['n_target']
    print("{:12s} {:>6d} {:>10d} {:>10d} {:>8.1f} {:>8s}".format(
        pop, info['n_case'], info['n_available'], info['n_target'], ratio, ok))
print("{:12s} {:>6d} {:>10s} {:>10d} {:>8.1f}".format(
    'TOTAL', total_case, '', total_target,
    total_target / total_case if total_case > 0 else 0))

design_rows = []
for pop in REPLICATION_POPS:
    info = pop_ctrl_map[pop]
    design_rows.append({
        'Population': pop, 'AD': info['n_case'],
        'CN_available': info['n_available'], 'CN_target': info['n_target'],
        'ratio': info['n_target'] / info['n_case'] if info['n_case'] > 0 else 0,
        'achievable': info['n_target'] == int(info['n_case'] * TARGET_RATIO)
    })
design_rows.append({
    'Population': 'TOTAL', 'AD': total_case,
    'CN_available': sum(i['n_available'] for i in pop_ctrl_map.values()),
    'CN_target': total_target,
    'ratio': total_target / total_case if total_case > 0 else 0,
    'achievable': True
})
pd.DataFrame(design_rows).to_csv('{}/Phase_C_downsampling_design.tsv'.format(OUT), sep='\t', index=False)


def compute_variant_stats(case_idx, ctrl_idx):
    case_genos = geno_matrix[case_idx, :]
    ctrl_genos = geno_matrix[ctrl_idx, :]
    case_ac = np.nansum(case_genos, axis=0).astype(int)
    ctrl_ac = np.nansum(ctrl_genos, axis=0).astype(int)
    case_an = (np.sum(~np.isnan(case_genos), axis=0) * 2).astype(int)
    ctrl_an = (np.sum(~np.isnan(ctrl_genos), axis=0) * 2).astype(int)
    total_ac = case_ac + ctrl_ac

    valid = (total_ac >= 3) & (case_an > 0) & (ctrl_an > 0)
    case_af = np.where(case_an > 0, case_ac / case_an, 0)
    ctrl_af = np.where(ctrl_an > 0, ctrl_ac / ctrl_an, 0)
    cc_ratio = np.where(ctrl_af > 0, case_af / ctrl_af, np.inf)
    valid = valid & np.isfinite(cc_ratio)

    rows = []
    for vi in np.where(valid)[0]:
        vk = variant_keys[vi]
        ag = vk_to_ag.get(vk, {})
        gene = vk_to_gene.get(vk, '')
        enrichment = 'case_enriched' if cc_ratio[vi] > 1 else 'control_enriched'
        if case_ac[vi] > 0 and ctrl_ac[vi] == 0:
            enrichment = 'case_only'
        rows.append({
            'variant_key': vk, 'gene_name': gene,
            'cc_ratio': cc_ratio[vi],
            'is_case_enriched': enrichment in ('case_enriched', 'case_only'),
            **ag
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


print("\n--- Full (unmatched) IR ---")
full_df = compute_variant_stats(case_indices, full_ctrl_indices)
print("Variants: {:,}, CE%: {:.1f}%".format(
    len(full_df), full_df['is_case_enriched'].mean() * 100 if len(full_df) > 0 else 0))
for mod in AG_MODALITIES:
    ir = calc_ir(full_df, mod, threshold_pct=80)
    if ir:
        print("  {:15s} IR={:.3f} P={:.2e}".format(AG_LABELS[mod], ir['IR'], ir['fisher_p']))


print("\n--- C-3. Bootstrap ({} iterations) ---".format(N_BOOTSTRAP))
np.random.seed(42)

boot_results = []
for b in range(N_BOOTSTRAP):
    sub_ctrl_iids = []
    for pop in REPLICATION_POPS:
        info = pop_ctrl_map[pop]
        if info['n_target'] > 0 and len(info['iids']) > 0:
            picked = np.random.choice(info['iids'], size=info['n_target'], replace=False)
            sub_ctrl_iids.extend(picked)

    sub_ctrl_indices = np.array([sample_idx[iid] for iid in sub_ctrl_iids if iid in sample_idx])
    boot_df = compute_variant_stats(case_indices, sub_ctrl_indices)
    if len(boot_df) < 50:
        continue

    boot_iter = {
        'iteration': b,
        'n_variants': len(boot_df),
        'n_cases': len(case_indices),
        'n_ctrls': len(sub_ctrl_indices),
        'ratio': len(sub_ctrl_indices) / len(case_indices),
        'ce_pct': boot_df['is_case_enriched'].mean() * 100
    }
    for mod in AG_MODALITIES:
        ir = calc_ir(boot_df, mod, threshold_pct=80)
        if ir:
            boot_iter['{}_IR'.format(AG_LABELS[mod])] = ir['IR']
            boot_iter['{}_OR'.format(AG_LABELS[mod])] = ir['OR']
            boot_iter['{}_P'.format(AG_LABELS[mod])] = ir['fisher_p']
        else:
            boot_iter['{}_IR'.format(AG_LABELS[mod])] = np.nan
    boot_results.append(boot_iter)

    if (b + 1) % 20 == 0:
        med = np.nanmedian([r.get('RNA-seq_IR', np.nan) for r in boot_results[-20:]])
        print("  Iter {}/{}: RNA-seq IR median(last 20)={:.3f}".format(b + 1, N_BOOTSTRAP, med))

boot_all = pd.DataFrame(boot_results)
print("\nValid iterations: {}".format(len(boot_all)))


print("\n--- C-3 (per-pop). Population-specific Bootstrap ---")

pop_boot_results = {}
for pop in REPLICATION_POPS:
    info = pop_ctrl_map[pop]
    pop_case_idx = pop_case_map[pop]

    if info['n_case'] < 20:
        print("\n{}: AD={}, too few for bootstrap".format(pop, info['n_case']))
        pop_boot_results[pop] = pd.DataFrame()
        continue

    np.random.seed(42)
    pop_boots = []
    n_target = min(int(info['n_case'] * TARGET_RATIO), len(info['iids']))

    for b in range(N_BOOTSTRAP):
        picked = np.random.choice(info['iids'], size=n_target, replace=False)
        sub_ctrl_idx = np.array([sample_idx[iid] for iid in picked if iid in sample_idx])
        bdf = compute_variant_stats(pop_case_idx, sub_ctrl_idx)
        if len(bdf) < 20:
            continue
        row = {
            'iteration': b, 'n_variants': len(bdf),
            'ce_pct': bdf['is_case_enriched'].mean() * 100,
            'ratio': len(sub_ctrl_idx) / len(pop_case_idx)
        }
        for mod in AG_MODALITIES:
            ir = calc_ir(bdf, mod, threshold_pct=80)
            if ir:
                row['{}_IR'.format(AG_LABELS[mod])] = ir['IR']
                row['{}_P'.format(AG_LABELS[mod])] = ir['fisher_p']
            else:
                row['{}_IR'.format(AG_LABELS[mod])] = np.nan
        pop_boots.append(row)

    pop_boot_results[pop] = pd.DataFrame(pop_boots)
    if len(pop_boots) > 0:
        pdf = pd.DataFrame(pop_boots)
        rna_vals = pdf['RNA-seq_IR'].dropna()
        print("\n{}: {} valid iterations".format(pop, len(pdf)))
        print("  CE%: {:.1f}% [{:.1f}-{:.1f}]".format(
            pdf['ce_pct'].median(), pdf['ce_pct'].quantile(0.025), pdf['ce_pct'].quantile(0.975)))
        if len(rna_vals) > 0:
            print("  RNA-seq IR: {:.3f} [{:.3f}-{:.3f}]".format(
                rna_vals.median(), rna_vals.quantile(0.025), rna_vals.quantile(0.975)))


print("\n" + "=" * 70)
print("PHASE C RESULTS: BOOTSTRAP IR (matched ~1:{:.1f})".format(TARGET_RATIO))
print("=" * 70)

print("\n{:15s} {:>8s} {:>8s} {:>20s} {:>8s} {:>8s} {:>10s}".format(
    'Modality', 'Mean', 'Median', '95% CI', 'R4 IR', 'InCI?', 'Direction'))
print("-" * 85)

summary_rows = []
for mod in AG_MODALITIES:
    col = '{}_IR'.format(AG_LABELS[mod])
    if col in boot_all.columns:
        vals = boot_all[col].dropna()
        if len(vals) > 0:
            r4_ir = R4_REF[AG_LABELS[mod]]['IR']
            ci_low = vals.quantile(0.025)
            ci_high = vals.quantile(0.975)
            in_ci = ci_low <= r4_ir <= ci_high
            same_dir = (r4_ir >= 1 and vals.median() >= 1) or (r4_ir < 1 and vals.median() < 1)
            print("{:15s} {:>8.3f} {:>8.3f} [{:.3f}-{:.3f}] {:>8.3f} {:>8s} {:>10s}".format(
                AG_LABELS[mod], vals.mean(), vals.median(),
                ci_low, ci_high, r4_ir,
                'YES' if in_ci else 'NO',
                'SAME' if same_dir else 'OPPOSITE'))
            summary_rows.append({
                'Modality': AG_LABELS[mod],
                'R5_mean': vals.mean(), 'R5_median': vals.median(),
                'R5_CI_low': ci_low, 'R5_CI_high': ci_high,
                'R4_IR': r4_ir, 'R4_in_CI': in_ci, 'Same_direction': same_dir
            })

ce_vals = boot_all['ce_pct'].dropna()
print("\nCE%: {:.1f}% [{:.1f}-{:.1f}] (R4: 71.3%)".format(
    ce_vals.median(), ce_vals.quantile(0.025), ce_vals.quantile(0.975)))
print("Matched ratio: {:.2f}".format(boot_all['ratio'].median()))

print("\n--- Per-Population Bootstrap IR (RNA-seq) ---")
print("{:12s} {:>8s} {:>20s} {:>8s} {:>8s}".format(
    'Population', 'Median', '95% CI', 'R4', 'InCI?'))
print("-" * 65)
pop_summary = []
for pop in REPLICATION_POPS:
    pdf = pop_boot_results.get(pop, pd.DataFrame())
    if len(pdf) > 0 and 'RNA-seq_IR' in pdf.columns:
        vals = pdf['RNA-seq_IR'].dropna()
        if len(vals) > 0:
            ci_l = vals.quantile(0.025)
            ci_h = vals.quantile(0.975)
            in_ci = ci_l <= 1.086 <= ci_h
            print("{:12s} {:>8.3f} [{:.3f}-{:.3f}] {:>8.3f} {:>8s}".format(
                pop, vals.median(), ci_l, ci_h, 1.086, 'YES' if in_ci else 'NO'))
            pop_summary.append({
                'Population': pop,
                'RNA_seq_median': vals.median(),
                'CI_low': ci_l, 'CI_high': ci_h,
                'CE_pct_median': pdf['ce_pct'].median(),
                'N_iterations': len(pdf)
            })
    else:
        print("{:12s} N/A (too few cases)".format(pop))


print("\n--- C-4. Before vs After Matching ---")
print("{:20s} {:>15s} {:>15s} {:>10s}".format('Metric', 'Unmatched', 'Matched(med)', 'R4'))
print("-" * 65)
if len(full_df) > 0:
    print("{:20s} {:>15.1f} {:>15.1f} {:>10.1f}".format(
        'CE%', full_df['is_case_enriched'].mean() * 100, ce_vals.median(), 71.3))
    print("{:20s} {:>15s} {:>15s} {:>10s}".format(
        'Ratio', '1:{:.1f}'.format(len(full_ctrl_indices) / len(case_indices)),
        '1:{:.1f}'.format(boot_all['ratio'].median()), '1:2.9'))
    for mod in AG_MODALITIES:
        full_ir = calc_ir(full_df, mod, threshold_pct=80)
        col = '{}_IR'.format(AG_LABELS[mod])
        r4_ir = R4_REF[AG_LABELS[mod]]['IR']
        if full_ir and col in boot_all.columns:
            matched_med = boot_all[col].dropna().median()
            print("{:20s} {:>15.3f} {:>15.3f} {:>10.3f}".format(
                AG_LABELS[mod] + ' IR', full_ir['IR'], matched_med, r4_ir))


boot_all.to_csv('{}/Phase_C_bootstrap_IR_all.tsv'.format(OUT), sep='\t', index=False)
pd.DataFrame(summary_rows).to_csv('{}/Phase_C_bootstrap_IR_summary.tsv'.format(OUT), sep='\t', index=False)

pop_boot_rows = []
for pop in REPLICATION_POPS:
    pdf = pop_boot_results.get(pop, pd.DataFrame())
    if len(pdf) > 0:
        pdf = pdf.copy()
        pdf['Population'] = pop
        pop_boot_rows.append(pdf)
if pop_boot_rows:
    pd.concat(pop_boot_rows, ignore_index=True).to_csv(
        '{}/Phase_C_bootstrap_IR_by_population.tsv'.format(OUT), sep='\t', index=False)

if len(full_df) > 0 and len(boot_all) > 0:
    ba_rows = []
    ba_rows.append({'Metric': 'CE%',
                    'Unmatched': full_df['is_case_enriched'].mean() * 100,
                    'Matched_median': ce_vals.median(), 'R4': 71.3})
    for mod in AG_MODALITIES:
        full_ir = calc_ir(full_df, mod, threshold_pct=80)
        col = '{}_IR'.format(AG_LABELS[mod])
        if full_ir and col in boot_all.columns:
            ba_rows.append({'Metric': AG_LABELS[mod] + '_IR',
                            'Unmatched': full_ir['IR'],
                            'Matched_median': boot_all[col].dropna().median(),
                            'R4': R4_REF[AG_LABELS[mod]]['IR']})
    pd.DataFrame(ba_rows).to_csv('{}/Phase_C_before_after_comparison.tsv'.format(OUT),
                                  sep='\t', index=False)

print("\nAll Phase C outputs saved.")
print("Finished: {}".format(datetime.now()))
