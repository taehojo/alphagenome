"""
Phase 7 Step 5: chromHMM Enrichment Analysis

Overlap ALL 9,943 variants (AC>=3) with Roadmap Epigenomics brain chromHMM states.
Compare chromatin state distributions between case-enriched vs control-enriched variants.
Fisher's exact test for each state × epigenome.

chromHMM: hg19 coordinates → liftOver variants from hg38 to hg19.
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import gzip
import json
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = '/N/project/AiLab/alphagenome'
VARIANT_FILE = f'{BASE}/data/variant_cc_with_alphgenome.csv'
CHROMHMM_DIR = f'{BASE}/data/chromhmm_brain'
CHAIN_FILE = f'{CHROMHMM_DIR}/hg38ToHg19.over.chain.gz'
LIFTOVER_BIN = '/N/project/AiLab/miniconda3/envs/liftover_env/bin/liftOver'
OUT_DIR = f'{BASE}/results/phase7_chromhmm'
os.makedirs(OUT_DIR, exist_ok=True)

EPIGENOMES = {
    'E067': 'Angular Gyrus',
    'E068': 'Anterior Caudate',
    'E069': 'Cingulate Gyrus',
    'E071': 'Hippocampus Middle',
    'E072': 'Inferior Temporal Lobe',
    'E073': 'DLPFC',
    'E074': 'Substantia Nigra',
    'E125': 'NH-A Astrocytes',
}

STATE_LABELS = {
    '1_TssA': 'Active TSS',
    '2_TssAFlnk': 'Flanking Active TSS',
    '3_TxFlnk': "Transcr. at gene 5'/3'",
    '4_Tx': 'Strong Transcription',
    '5_TxWk': 'Weak Transcription',
    '6_EnhG': 'Genic Enhancers',
    '7_Enh': 'Enhancers',
    '8_ZNF/Rpts': 'ZNF Genes & Repeats',
    '9_Het': 'Heterochromatin',
    '10_TssBiv': 'Bivalent/Poised TSS',
    '11_BivFlnk': 'Flanking Bivalent TSS/Enh',
    '12_EnhBiv': 'Bivalent Enhancer',
    '13_ReprPC': 'Repressed PolyComb',
    '14_ReprPCWk': 'Weak Repressed PolyComb',
    '15_Quies': 'Quiescent/Low',
}

REGULATORY_STATES = ['1_TssA', '2_TssAFlnk', '3_TxFlnk', '6_EnhG', '7_Enh',
                     '10_TssBiv', '11_BivFlnk', '12_EnhBiv']
ACTIVE_REGULATORY = ['1_TssA', '2_TssAFlnk', '6_EnhG', '7_Enh']
TRANSCRIPTION_STATES = ['4_Tx', '5_TxWk']
REPRESSIVE_STATES = ['9_Het', '13_ReprPC', '14_ReprPCWk']

def load_and_prep_variants():
    """Load variants, apply AC>=3 filter, deduplicate."""
    df = pd.read_csv(VARIANT_FILE)
    df['total_AC'] = df['case_AC'] + df['ctrl_AC']
    df = df[df['total_AC'] >= 3]
    df = df.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')

    
    if 'enrichment' not in df.columns:
        df['case_AF'] = df['case_AC'] / df['case_AN']
        df['ctrl_AF'] = df['ctrl_AC'] / df['ctrl_AN']
        df['cc_ratio'] = df['case_AF'] / df['ctrl_AF']
        df['enrichment'] = np.where(df['cc_ratio'] > 1, 'case', 'control')
    
    df['enrichment'] = df['enrichment'].str.lower().str.strip()
    df.loc[df['enrichment'].str.contains('case', na=False), 'enrichment'] = 'case'
    df.loc[~df['enrichment'].str.contains('case', na=False), 'enrichment'] = 'control'

    print(f"Loaded {len(df)} unique variants (AC>=3)")
    print(f"  Case-enriched: {(df['enrichment']=='case').sum()}")
    print(f"  Control-enriched: {(df['enrichment']=='control').sum()}")
    return df

def liftover_variants(df):
    """LiftOver variant positions from hg38 to hg19."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
        bed_in = f.name
        for _, row in df.iterrows():
            chrom = f"chr{int(row['chr_num'])}"
            pos = int(row['pos'])
            f.write(f"{chrom}\t{pos-1}\t{pos}\t{row['variant_id']}\n")

    bed_out = bed_in.replace('.bed', '_hg19.bed')
    unmapped = bed_in.replace('.bed', '_unmapped.bed')

    cmd = [LIFTOVER_BIN, bed_in, CHAIN_FILE, bed_out, unmapped]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    
    mapped = {}
    with open(bed_out) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom, start, end, vid = parts[0], int(parts[1]), int(parts[2]), parts[3]
            mapped[vid] = (chrom, start, end)

    
    n_unmapped = 0
    with open(unmapped) as f:
        for line in f:
            if not line.startswith('#'):
                n_unmapped += 1

    print(f"LiftOver: {len(mapped)} mapped, {n_unmapped} unmapped")

    
    os.unlink(bed_in)
    os.unlink(bed_out)
    os.unlink(unmapped)

    return mapped

def load_chromhmm(epigenome_id):
    """Load chromHMM BED file (gzipped)."""
    filepath = f"{CHROMHMM_DIR}/{epigenome_id}_15_coreMarks_dense.bed.gz"
    regions = []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                state = parts[3]
                regions.append((chrom, start, end, state))
    return regions

def build_interval_index(regions):
    """Build a dict of chrom -> sorted list of (start, end, state) for binary search."""
    by_chrom = defaultdict(list)
    for chrom, start, end, state in regions:
        by_chrom[chrom].append((start, end, state))
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    return by_chrom

def lookup_state(interval_index, chrom, pos):
    """Find chromHMM state for a position using binary search."""
    if chrom not in interval_index:
        return None
    regions = interval_index[chrom]
    lo, hi = 0, len(regions) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, state = regions[mid]
        if pos < start:
            hi = mid - 1
        elif pos >= end:
            lo = mid + 1
        else:
            return state
    return None

def run_enrichment_analysis():
    """Main analysis: overlap variants with chromHMM, compute enrichment."""
    
    df = load_and_prep_variants()

    
    mapped = liftover_variants(df)
    df_mapped = df[df['variant_id'].isin(mapped)].copy()
    df_mapped['hg19_chrom'] = df_mapped['variant_id'].map(lambda x: mapped[x][0])
    df_mapped['hg19_pos'] = df_mapped['variant_id'].map(lambda x: mapped[x][1])
    print(f"Working with {len(df_mapped)} variants after liftOver")

    
    all_state_results = {}

    for eid, ename in EPIGENOMES.items():
        print(f"\nProcessing {eid} ({ename})...")
        regions = load_chromhmm(eid)
        idx = build_interval_index(regions)

        states = []
        for _, row in df_mapped.iterrows():
            state = lookup_state(idx, row['hg19_chrom'], row['hg19_pos'])
            states.append(state)

        df_mapped[f'state_{eid}'] = states
        annotated = df_mapped[df_mapped[f'state_{eid}'].notna()]
        print(f"  Annotated: {len(annotated)}/{len(df_mapped)} variants")

        
        case_states = annotated[annotated['enrichment'] == 'case'][f'state_{eid}'].value_counts()
        ctrl_states = annotated[annotated['enrichment'] == 'control'][f'state_{eid}'].value_counts()

        n_case = (annotated['enrichment'] == 'case').sum()
        n_ctrl = (annotated['enrichment'] == 'control').sum()

        
        state_results = []
        all_states = sorted(set(case_states.index) | set(ctrl_states.index))
        for state in all_states:
            a = case_states.get(state, 0)  
            b = n_case - a                  
            c = ctrl_states.get(state, 0)   
            d = n_ctrl - c                  

            odds_ratio, pval = stats.fisher_exact([[a, b], [c, d]])

            state_results.append({
                'epigenome': eid,
                'epigenome_name': ename,
                'state': state,
                'state_label': STATE_LABELS.get(state, state),
                'case_count': a,
                'case_pct': a / n_case * 100 if n_case > 0 else 0,
                'ctrl_count': c,
                'ctrl_pct': c / n_ctrl * 100 if n_ctrl > 0 else 0,
                'odds_ratio': odds_ratio,
                'pvalue': pval,
                'n_case_total': n_case,
                'n_ctrl_total': n_ctrl,
            })

        all_state_results[eid] = state_results

    
    results_list = []
    for eid in all_state_results:
        results_list.extend(all_state_results[eid])
    results_df = pd.DataFrame(results_list)

    
    from statsmodels.stats.multitest import multipletests
    _, results_df['fdr'], _, _ = multipletests(results_df['pvalue'], method='fdr_bh')

    results_df.to_csv(f'{OUT_DIR}/chromhmm_enrichment_all.csv', index=False)
    print(f"\nSaved full results: {len(results_df)} state × epigenome tests")

    
    grouped_results = []
    for eid, ename in EPIGENOMES.items():
        col = f'state_{eid}'
        annotated = df_mapped[df_mapped[col].notna()]

        for group_name, group_states in [
            ('Active Regulatory', ACTIVE_REGULATORY),
            ('All Regulatory', REGULATORY_STATES),
            ('Transcription', TRANSCRIPTION_STATES),
            ('Repressive', REPRESSIVE_STATES),
        ]:
            case_in = annotated[(annotated['enrichment'] == 'case') & (annotated[col].isin(group_states))].shape[0]
            case_out = annotated[(annotated['enrichment'] == 'case') & (~annotated[col].isin(group_states))].shape[0]
            ctrl_in = annotated[(annotated['enrichment'] == 'control') & (annotated[col].isin(group_states))].shape[0]
            ctrl_out = annotated[(annotated['enrichment'] == 'control') & (~annotated[col].isin(group_states))].shape[0]

            odds_ratio, pval = stats.fisher_exact([[case_in, case_out], [ctrl_in, ctrl_out]])

            grouped_results.append({
                'epigenome': eid,
                'epigenome_name': ename,
                'state_group': group_name,
                'case_in': case_in,
                'case_pct': case_in / (case_in + case_out) * 100,
                'ctrl_in': ctrl_in,
                'ctrl_pct': ctrl_in / (ctrl_in + ctrl_out) * 100,
                'odds_ratio': odds_ratio,
                'pvalue': pval,
            })

    grouped_df = pd.DataFrame(grouped_results)
    _, grouped_df['fdr'], _, _ = multipletests(grouped_df['pvalue'], method='fdr_bh')
    grouped_df.to_csv(f'{OUT_DIR}/chromhmm_grouped_enrichment.csv', index=False)

    
    
    reg_count = pd.Series(0, index=df_mapped.index)
    active_reg_count = pd.Series(0, index=df_mapped.index)
    for eid in EPIGENOMES:
        col = f'state_{eid}'
        reg_count += df_mapped[col].isin(REGULATORY_STATES).astype(int)
        active_reg_count += df_mapped[col].isin(ACTIVE_REGULATORY).astype(int)

    df_mapped['n_regulatory_epigenomes'] = reg_count
    df_mapped['n_active_regulatory_epigenomes'] = active_reg_count

    
    case_reg = df_mapped[df_mapped['enrichment'] == 'case']['n_regulatory_epigenomes']
    ctrl_reg = df_mapped[df_mapped['enrichment'] == 'control']['n_regulatory_epigenomes']
    u_stat, u_pval = stats.mannwhitneyu(case_reg, ctrl_reg, alternative='greater')

    case_areg = df_mapped[df_mapped['enrichment'] == 'case']['n_active_regulatory_epigenomes']
    ctrl_areg = df_mapped[df_mapped['enrichment'] == 'control']['n_active_regulatory_epigenomes']
    u_stat2, u_pval2 = stats.mannwhitneyu(case_areg, ctrl_areg, alternative='greater')

    consensus = {
        'regulatory_case_mean': case_reg.mean(),
        'regulatory_ctrl_mean': ctrl_reg.mean(),
        'regulatory_mannwhitney_U': u_stat,
        'regulatory_pvalue': u_pval,
        'active_reg_case_mean': case_areg.mean(),
        'active_reg_ctrl_mean': ctrl_areg.mean(),
        'active_reg_mannwhitney_U': u_stat2,
        'active_reg_pvalue': u_pval2,
    }

    print(f"\n--- Cross-epigenome consensus ---")
    print(f"Regulatory states (any): case={case_reg.mean():.3f} vs ctrl={ctrl_reg.mean():.3f}, P={u_pval:.4e}")
    print(f"Active regulatory: case={case_areg.mean():.3f} vs ctrl={ctrl_areg.mean():.3f}, P={u_pval2:.4e}")

    
    save_cols = ['variant_id', 'gene_name', 'chr_num', 'pos', 'enrichment', 'cc_ratio',
                 'n_regulatory_epigenomes', 'n_active_regulatory_epigenomes']
    for eid in EPIGENOMES:
        save_cols.append(f'state_{eid}')
    df_mapped[save_cols].to_csv(f'{OUT_DIR}/variants_chromhmm_annotated.csv', index=False)

    return results_df, grouped_df, consensus, df_mapped

def create_figures(results_df, grouped_df, consensus, df_mapped):
    """Create visualization figures."""

    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    
    pivot_case = results_df.pivot_table(index='state', columns='epigenome_name', values='case_pct')
    pivot_ctrl = results_df.pivot_table(index='state', columns='epigenome_name', values='ctrl_pct')

    
    state_order = sorted(pivot_case.index, key=lambda x: int(x.split('_')[0]))
    pivot_case = pivot_case.reindex(state_order)
    pivot_ctrl = pivot_ctrl.reindex(state_order)

    
    rename_map = {s: f"{s.split('_')[0]}. {STATE_LABELS.get(s, s)}" for s in state_order}
    pivot_case.index = [rename_map[s] for s in pivot_case.index]
    pivot_ctrl.index = [rename_map[s] for s in pivot_ctrl.index]

    sns.heatmap(pivot_case, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0],
                cbar_kws={'label': '% of variants'})
    axes[0].set_title('Case-Enriched Variants\n(cc_ratio > 1)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('')

    sns.heatmap(pivot_ctrl, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': '% of variants'})
    axes[1].set_title('Control-Enriched Variants\n(cc_ratio < 1)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_chromhmm_state_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()

    
    fig, ax = plt.subplots(figsize=(10, 8))

    
    ar_df = grouped_df[grouped_df['state_group'] == 'Active Regulatory'].sort_values('odds_ratio')

    y_pos = range(len(ar_df))
    colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in ar_df['fdr']]

    ax.barh(y_pos, ar_df['odds_ratio'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['epigenome_name']}\n(P={row['pvalue']:.3f})" for _, row in ar_df.iterrows()])
    ax.set_xlabel('Odds Ratio (Case-enriched vs Control-enriched)', fontsize=12)
    ax.set_title('Active Regulatory State Enrichment\nin Case-Enriched AD Variants', fontsize=14, fontweight='bold')

    
    ax.text(0.98, 0.02, f"n_case={ar_df.iloc[0]['case_in'] + int(ar_df.iloc[0]['case_pct']/100*(ar_df.iloc[0]['case_in']/(ar_df.iloc[0]['case_pct']/100)))}"
            if len(ar_df) > 0 else "", transform=ax.transAxes, ha='right', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_active_regulatory_OR.png', dpi=200, bbox_inches='tight')
    plt.close()

    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    case_data = df_mapped[df_mapped['enrichment'] == 'case']['n_regulatory_epigenomes']
    ctrl_data = df_mapped[df_mapped['enrichment'] == 'control']['n_regulatory_epigenomes']

    bins = np.arange(-0.5, 9.5, 1)
    axes[0].hist(case_data, bins=bins, alpha=0.7, color='#e74c3c', label=f'Case (n={len(case_data)})', density=True)
    axes[0].hist(ctrl_data, bins=bins, alpha=0.7, color='#3498db', label=f'Control (n={len(ctrl_data)})', density=True)
    axes[0].set_xlabel('Number of Epigenomes with Regulatory State', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('All Regulatory States', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].text(0.95, 0.95, f"P = {consensus['regulatory_pvalue']:.2e}",
                transform=axes[0].transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    case_data2 = df_mapped[df_mapped['enrichment'] == 'case']['n_active_regulatory_epigenomes']
    ctrl_data2 = df_mapped[df_mapped['enrichment'] == 'control']['n_active_regulatory_epigenomes']

    axes[1].hist(case_data2, bins=bins, alpha=0.7, color='#e74c3c', label=f'Case (n={len(case_data2)})', density=True)
    axes[1].hist(ctrl_data2, bins=bins, alpha=0.7, color='#3498db', label=f'Control (n={len(ctrl_data2)})', density=True)
    axes[1].set_xlabel('Number of Epigenomes with Active Regulatory State', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Active Regulatory States (TSS, Enhancer)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].text(0.95, 0.95, f"P = {consensus['active_reg_pvalue']:.2e}",
                transform=axes[1].transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_regulatory_count_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()

    
    fig, ax = plt.subplots(figsize=(12, 8))

    pivot_or = results_df.pivot_table(index='state', columns='epigenome_name', values='odds_ratio')
    pivot_pval = results_df.pivot_table(index='state', columns='epigenome_name', values='pvalue')

    state_order = sorted(pivot_or.index, key=lambda x: int(x.split('_')[0]))
    pivot_or = pivot_or.reindex(state_order)
    pivot_pval = pivot_pval.reindex(state_order)

    rename_map = {s: f"{s.split('_')[0]}. {STATE_LABELS.get(s, s)}" for s in state_order}
    pivot_or.index = [rename_map[s] for s in state_order]
    pivot_pval_renamed = pivot_pval.copy()
    pivot_pval_renamed.index = [rename_map[s] for s in state_order]

    
    log2_or = np.log2(pivot_or.replace(0, np.nan).replace(np.inf, np.nan))

    
    annot = pd.DataFrame('', index=log2_or.index, columns=log2_or.columns)
    for i in log2_or.index:
        for j in log2_or.columns:
            val = pivot_or.loc[i, j] if i in pivot_or.index else np.nan
            pv = pivot_pval_renamed.loc[i, j] if i in pivot_pval_renamed.index else np.nan
            if pd.notna(val) and pd.notna(pv):
                star = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
                annot.loc[i, j] = f'{val:.2f}{star}'

    sns.heatmap(log2_or, annot=annot, fmt='', cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'log2(Odds Ratio)'}, linewidths=0.5)
    ax.set_title('Chromatin State Enrichment in Case-Enriched Variants\n(Odds Ratio, * P<0.05, ** P<0.01, *** P<0.001)',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_OR_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {OUT_DIR}/")

def main():
    print("=" * 60)
    print("STEP 5: chromHMM Enrichment Analysis")
    print("=" * 60)

    results_df, grouped_df, consensus, df_mapped = run_enrichment_analysis()
    create_figures(results_df, grouped_df, consensus, df_mapped)

    
    sig_states = results_df[results_df['fdr'] < 0.05]
    sig_grouped = grouped_df[grouped_df['fdr'] < 0.05]

    summary = {
        'step': 5,
        'status': 'SUCCESS',
        'n_variants_analyzed': len(df_mapped),
        'n_epigenomes': len(EPIGENOMES),
        'total_tests': len(results_df),
        'significant_state_tests_fdr05': len(sig_states),
        'significant_grouped_tests_fdr05': len(sig_grouped),
        'consensus': consensus,
        'top_findings': [],
    }

    
    for _, row in grouped_df.sort_values('pvalue').head(5).iterrows():
        summary['top_findings'].append({
            'epigenome': row['epigenome_name'],
            'state_group': row['state_group'],
            'OR': round(row['odds_ratio'], 4),
            'pvalue': row['pvalue'],
            'fdr': row['fdr'],
            'case_pct': round(row['case_pct'], 2),
            'ctrl_pct': round(row['ctrl_pct'], 2),
        })

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Variants analyzed: {summary['n_variants_analyzed']}")
    print(f"Significant state × epigenome tests (FDR<0.05): {summary['significant_state_tests_fdr05']}")
    print(f"Significant grouped tests (FDR<0.05): {summary['significant_grouped_tests_fdr05']}")
    print(f"\nCross-epigenome regulatory overlap:")
    print(f"  Case mean: {consensus['regulatory_case_mean']:.3f}")
    print(f"  Control mean: {consensus['regulatory_ctrl_mean']:.3f}")
    print(f"  P-value: {consensus['regulatory_pvalue']:.4e}")
    print(f"\nTop 5 grouped findings:")
    for f in summary['top_findings']:
        print(f"  {f['epigenome']} / {f['state_group']}: OR={f['OR']:.3f}, P={f['pvalue']:.4e}, FDR={f['fdr']:.4e}")

    
    log_path = f'{BASE}/results/phase7_logs/step_5.json'
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(log_path, 'w') as f:
        json.dump(summary, f, indent=2, default=convert)

    print(f"\nLog saved: {log_path}")
    return summary

if __name__ == '__main__':
    main()
