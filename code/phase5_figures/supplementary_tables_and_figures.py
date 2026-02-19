import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_RED = '#D55E00'
CB_GREEN = '#009E73'
CB_PURPLE = '#CC79A7'
CB_GRAY = '#999999'
CB_LIGHTBLUE = '#56B4E9'

BASE = '<WORK_DIR>'
R5_DIR = os.path.join(BASE, 'analysis/r5_replication')
ADD_DIR = os.path.join(BASE, 'analysis/additional_analyses')
OUT_DIR = os.path.join(BASE, 'analysis/manuscript_revision')

def mm2in(mm):
    return mm / 25.4


def create_table6():
    print("=== Creating Table 6: R5 Replication ===")

    summary = pd.read_csv(os.path.join(R5_DIR, 'Phase_C_bootstrap_IR_summary.tsv'), sep='\t')

    rows = []
    for _, r in summary.iterrows():
        mod = r['Modality']
        rows.append({
            'Modality': mod,
            'R4 Discovery IR': f"{r['R4_IR']:.3f}",
            'R5 Replication IR (Median)': f"{r['R5_median']:.3f}",
            'R5 Bootstrap 95% CI': f"[{r['R5_CI_low']:.3f}, {r['R5_CI_high']:.3f}]",
            'R4 within R5 CI?': 'Yes' if r['R4_in_CI'] else 'No',
        })

    boot_all = pd.read_csv(os.path.join(R5_DIR, 'Phase_C_bootstrap_IR_all.tsv'), sep='\t')
    for i, mod in enumerate(['RNA-seq', 'CAGE', 'DNase', 'ChIP-histone']):
        col = f"{mod}_P"
        if col in boot_all.columns:
            median_p = boot_all[col].median()
            rows[i]['R5 P-value (median bootstrap)'] = f"{median_p:.2e}"
        else:
            pass

    df_out = pd.DataFrame(rows)

    df_out.to_csv(os.path.join(OUT_DIR, 'Table6_R5_replication.tsv'), sep='\t', index=False)

    with open(os.path.join(OUT_DIR, 'Table6_R5_replication_formatted.csv'), 'w') as f:
        f.write("Table 6. Independent Replication in ADSP R5\n")
        f.write("\n")
        df_out.to_csv(f, index=False)
        f.write("\n")
        f.write("Note: R5-only cohort: N = 11545 (AD 1408 / CN 10137) from NHW / Hispanic / AA populations\n")
        f.write("with no overlap with R4. Case/control ratio matched to 1:2.9 via population-proportional\n")
        f.write("downsampling (100 bootstrap iterations). R4 Discovery IR from Table 3 (primary analysis).\n")

    print(f"  Saved: Table6_R5_replication.tsv")
    print(f"  Saved: Table6_R5_replication_formatted.csv")
    return df_out


def create_table7():
    print("=== Creating Table 7: Sensitivity Summary ===")

    ac_df = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_sensitivity_AC.tsv'), sep='\t')
    pct_df = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_sensitivity_percentile.tsv'), sep='\t')
    conf_df = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_confounders_correlation.tsv'), sep='\t')

    rna_ac = ac_df[ac_df['Modality'] == 'rna_seq'].copy()
    panel_a = []
    for _, r in rna_ac.iterrows():
        primary = ' *' if r['AC_threshold'] == 3 else ''
        panel_a.append({
            'Panel': 'A',
            'AC Threshold': f">= {int(r['AC_threshold'])}{primary}",
            'N variants': int(r['N_variants']),
            'CE%': f"{r['CE_pct']:.1f}%",
            'RNA-seq IR': f"{r['IR']:.3f}",
            'P-value': f"{r['P_value']:.2e}",
        })

    rna_pct = pct_df[pct_df['Modality'] == 'rna_seq'].drop_duplicates('Threshold_percentile').sort_values('Threshold_percentile')
    chip_pct = pct_df[pct_df['Modality'] == 'chip_histone'].drop_duplicates('Threshold_percentile').sort_values('Threshold_percentile')

    rna_dict = {int(r['Threshold_percentile']): r for _, r in rna_pct.iterrows()}
    chip_dict = {int(r['Threshold_percentile']): r for _, r in chip_pct.iterrows()}

    panel_b = []
    for thr in sorted(rna_dict.keys()):
        primary = ' *' if thr == 80 else ''
        top_pct = 100 - thr
        rr = rna_dict[thr]
        cr = chip_dict.get(thr, None)
        row = {
            'Panel': 'B',
            'Threshold': f"Top {top_pct}% (P{thr}){primary}",
            'RNA-seq IR': f"{rr['IR']:.3f}",
            'RNA-seq P': f"{rr['P_value']:.2e}",
        }
        if cr is not None:
            row['ChIP-histone IR'] = f"{cr['IR']:.3f}"
            row['ChIP-histone P'] = f"{cr['P_value']:.2e}"
        panel_b.append(row)

    panel_c = []
    for _, r in conf_df.iterrows():
        if r['Confounder'] in ['N_variants', 'Gene_length', 'Variant_density']:
            name_map = {'N_variants': 'Variant count', 'Gene_length': 'Gene length', 'Variant_density': 'Variant density'}
            panel_c.append({
                'Panel': 'C',
                'Variable': name_map.get(r['Confounder'], r['Confounder']),
                'Spearman r': f"{r['Spearman_r']:.3f}",
                'P-value': f"{r['P_value']:.2f}",
            })

    with open(os.path.join(OUT_DIR, 'Table7_sensitivity.tsv'), 'w') as f:
        f.write("# Table 7. Sensitivity Analysis Summary\n")
        f.write("\n# Panel A: AC Threshold Sensitivity (RNA-seq)\n")
        pd.DataFrame(panel_a).to_csv(f, sep='\t', index=False)
        f.write("\n# Panel B: Percentile Threshold Sensitivity\n")
        pd.DataFrame(panel_b).to_csv(f, sep='\t', index=False)
        f.write("\n# Panel C: Confounder Non-Correlation (Gene-level IR)\n")
        pd.DataFrame(panel_c).to_csv(f, sep='\t', index=False)

    with open(os.path.join(OUT_DIR, 'Table7_sensitivity_formatted.csv'), 'w') as f:
        f.write("Table 7. Sensitivity Analysis Summary\n\n")
        f.write("Panel A. AC Threshold Sensitivity (RNA-seq)\n")
        pd.DataFrame(panel_a).drop(columns=['Panel']).to_csv(f, index=False)
        f.write("\nPanel B. Percentile Threshold Sensitivity\n")
        pd.DataFrame(panel_b).drop(columns=['Panel']).to_csv(f, index=False)
        f.write("\nPanel C. Confounder Non-Correlation (Gene-level IR vs RNA-seq IR)\n")
        pd.DataFrame(panel_c).drop(columns=['Panel']).to_csv(f, index=False)
        f.write("\nNote: IR > 1 across all AC thresholds and 10/10 percentile thresholds for RNA-seq.\n")
        f.write("Decile dose-response: Spearman r = 0.784, P = 0.007.\n")
        f.write("No gene-level confounders correlate with IR (* = primary analysis threshold).\n")

    print(f"  Saved: Table7_sensitivity.tsv")
    print(f"  Saved: Table7_sensitivity_formatted.csv")


def create_table_s9():
    print("=== Creating Table S9: R5 Demographics ===")

    demo = pd.read_csv(os.path.join(R5_DIR, 'Phase_A_replication_cohort_demographics.tsv'), sep='\t')

    rows = []
    for _, r in demo.iterrows():
        pop = r['Population']
        rows.append({
            'Population': pop,
            'R5 AD': int(r['R5_AD']),
            'R5 CN': int(r['R5_CN']),
            'R5 Total': int(r['R5_Total']),
            'Case:Control Ratio': f"1:{r['R5_ratio']:.1f}",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, 'TableS9_R5_demographics.tsv'), sep='\t', index=False)

    print(f"  Saved: TableS9_R5_demographics.tsv")


def create_table_s10():
    print("=== Creating Table S10: R5 Per-Population ===")

    by_pop = pd.read_csv(os.path.join(R5_DIR, 'Phase_B_by_population.tsv'), sep='\t')
    demo = pd.read_csv(os.path.join(R5_DIR, 'Phase_A_replication_cohort_demographics.tsv'), sep='\t')

    boot_pop = pd.read_csv(os.path.join(R5_DIR, 'Phase_C_bootstrap_IR_by_population.tsv'), sep='\t')

    rows = []
    for _, r in by_pop.iterrows():
        pop = r['Population']
        d = demo[demo['Population'] == pop].iloc[0] if pop in demo['Population'].values else None

        bp = boot_pop[boot_pop['Population'] == pop]
        if len(bp) > 0:
            rna_vals = bp['RNA-seq_IR'].dropna()
            rna_median = rna_vals.median()
            rna_ci_lo = rna_vals.quantile(0.025)
            rna_ci_hi = rna_vals.quantile(0.975)
            matched_str = f"{rna_median:.3f}"
            ci_str = f"[{rna_ci_lo:.3f}, {rna_ci_hi:.3f}]"
            r4_in_ci = 'Yes' if rna_ci_lo <= 1.086 <= rna_ci_hi else 'No'
        else:
            matched_str = 'N/A'
            ci_str = 'N/A'
            r4_in_ci = 'N/A'

        row = {
            'Population': pop,
            'N': int(r['N']),
            'N AD': int(d['R5_AD']) if d is not None else '',
            'N CN': int(d['R5_CN']) if d is not None else '',
            'Unmatched RNA-seq IR': f"{r['RNA-seq_IR']:.3f}",
            'Unmatched P': f"{r['RNA-seq_P']:.2e}",
            'Matched RNA-seq IR (Median)': matched_str,
            'Bootstrap 95% CI': ci_str,
            'R4 IR within CI?': r4_in_ci,
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, 'TableS10_R5_per_population.tsv'), sep='\t', index=False)

    print(f"  Saved: TableS10_R5_per_population.tsv")


def create_table_s11():
    print("=== Creating Table S11: Gene IR Bootstrap ===")

    r4_boot = pd.read_csv(os.path.join(ADD_DIR, 'prompt6_gene_IR_bootstrap_R4.tsv'), sep='\t')
    r5_boot = pd.read_csv(os.path.join(ADD_DIR, 'prompt6_gene_IR_bootstrap_R5.tsv'), sep='\t')
    shrinkage = pd.read_csv(os.path.join(ADD_DIR, 'prompt6_shrinkage_IR.tsv'), sep='\t')

    r4_rna = r4_boot[r4_boot['modality'] == 'rna_seq'].copy()
    r5_rna = r5_boot[r5_boot['modality'] == 'rna_seq'].copy()

    merged = r4_rna.rename(columns={
        'IR_original': 'R4_IR', 'IR_mean': 'R4_IR_mean', 'IR_median': 'R4_IR_median',
        'CI_lower': 'R4_CI_lower', 'CI_upper': 'R4_CI_upper',
        'includes_1': 'R4_includes_1', 'n_variants': 'N_variants'
    })[['gene', 'N_variants', 'R4_IR', 'R4_IR_median', 'R4_CI_lower', 'R4_CI_upper', 'R4_includes_1']]

    r5_cols = r5_rna.rename(columns={
        'IR_original': 'R5_IR', 'IR_median': 'R5_IR_median',
        'CI_lower': 'R5_CI_lower', 'CI_upper': 'R5_CI_upper',
        'includes_1': 'R5_includes_1'
    })[['gene', 'R5_IR', 'R5_IR_median', 'R5_CI_lower', 'R5_CI_upper', 'R5_includes_1']]

    merged = merged.merge(r5_cols, on='gene', how='left')

    shrink_rna = shrinkage[shrinkage['modality'] == 'rna_seq'][['gene', 'IR_shrunken']].copy()
    shrink_rna = shrink_rna.rename(columns={'IR_shrunken': 'shrinkage_IR'})
    merged = merged.merge(shrink_rna, on='gene', how='left')

    merged['R4_in_R5_CI'] = merged.apply(
        lambda r: 'Yes' if pd.notna(r.get('R5_CI_lower')) and r['R5_CI_lower'] <= r['R4_IR'] <= r['R5_CI_upper'] else
                  ('No' if pd.notna(r.get('R5_CI_lower')) else 'N/A'), axis=1)

    merged = merged.sort_values('R4_IR', ascending=False)

    out_rows = []
    for _, r in merged.iterrows():
        row = {
            'Gene': r['gene'],
            'N variants': int(r['N_variants']),
            'R4 IR': f"{r['R4_IR']:.3f}",
            'R4 95% CI': f"[{r['R4_CI_lower']:.3f}, {r['R4_CI_upper']:.3f}]",
            'R4 CI excludes 1': 'Yes' if not r['R4_includes_1'] else 'No',
            'Shrinkage IR': f"{r['shrinkage_IR']:.3f}" if pd.notna(r.get('shrinkage_IR')) else 'N/A',
        }
        if pd.notna(r.get('R5_IR')):
            row['R5 IR'] = f"{r['R5_IR']:.3f}"
            row['R5 95% CI'] = f"[{r['R5_CI_lower']:.3f}, {r['R5_CI_upper']:.3f}]"
            row['R4 in R5 CI?'] = r['R4_in_R5_CI']
        else:
            row['R5 IR'] = 'N/A'
            row['R5 95% CI'] = 'N/A'
            row['R4 in R5 CI?'] = 'N/A'
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(os.path.join(OUT_DIR, 'TableS11_gene_IR_bootstrap.tsv'), sep='\t', index=False)

    print(f"  Saved: TableS11_gene_IR_bootstrap.tsv ({len(df_out)} genes)")


def create_table_s12():
    print("=== Creating Table S12: Regression ===")

    firth = pd.read_csv(os.path.join(ADD_DIR, 'prompt7_firth_R4.tsv'), sep='\t')
    vs_cc = pd.read_csv(os.path.join(ADD_DIR, 'prompt7_firth_vs_ccratio.tsv'), sep='\t')
    corr = pd.read_csv(os.path.join(ADD_DIR, 'prompt7_modality_correlation.tsv'), sep='\t')
    std = pd.read_csv(os.path.join(ADD_DIR, 'prompt7_modality_standardization.tsv'), sep='\t')

    with open(os.path.join(OUT_DIR, 'TableS12_regression.tsv'), 'w') as f:
        f.write("# Supplementary Table S12. Regression Analysis Results\n\n")

        f.write("# Panel A: Single-Modality Logistic Regression (R4)\n")
        r4_single = firth[(firth['Cohort'] == 'R4') & (firth['Model'] == 'single')]
        panel_a = []
        for _, r in r4_single.iterrows():
            mod = r['Modality']
            skew_row = std[std['Modality'] == mod]
            skewness = f"{skew_row['Skewness'].values[0]:.1f}" if len(skew_row) > 0 else 'N/A'

            cc_row = vs_cc[vs_cc['Modality'] == mod]
            cc_ir = f"{cc_row['CC_IR'].values[0]:.3f}" if len(cc_row) > 0 else 'N/A'
            cc_p = f"{cc_row['CC_Fisher_P'].values[0]:.2e}" if len(cc_row) > 0 else 'N/A'

            panel_a.append({
                'Modality': mod,
                'OR': f"{r['OR']:.3f}",
                '95% CI': f"[{r['OR_CI_lower']:.3f}, {r['OR_CI_upper']:.3f}]",
                'P-value': f"{r['P_value']:.2e}" if r['P_value'] < 0.01 else f"{r['P_value']:.3f}",
                'Skewness': skewness,
                'CC-ratio IR': cc_ir,
                'CC-ratio P': cc_p,
            })
        pd.DataFrame(panel_a).to_csv(f, sep='\t', index=False)

        f.write("\n# Panel B: Multi-Modality Logistic Regression (R4)\n")
        r4_multi = firth[(firth['Cohort'] == 'R4') & (firth['Model'] == 'multi')]
        panel_b = []
        for _, r in r4_multi.iterrows():
            panel_b.append({
                'Variable': r['Modality'],
                'OR': f"{r['OR']:.3f}",
                '95% CI': f"[{r['OR_CI_lower']:.3f}, {r['OR_CI_upper']:.3f}]",
                'P-value': f"{r['P_value']:.2e}" if r['P_value'] < 0.01 else f"{r['P_value']:.3f}",
            })
        pd.DataFrame(panel_b).to_csv(f, sep='\t', index=False)

        f.write("\n# Panel C: R5 Single-Modality Regression\n")
        r5_single = firth[(firth['Cohort'] == 'R5') & (firth['Model'] == 'single')]
        panel_c = []
        for _, r in r5_single.iterrows():
            panel_c.append({
                'Modality': r['Modality'],
                'R5 OR': f"{r['OR']:.3f}",
                '95% CI': f"[{r['OR_CI_lower']:.3f}, {r['OR_CI_upper']:.3f}]",
                'R5 P-value': f"{r['P_value']:.2e}" if r['P_value'] < 0.01 else f"{r['P_value']:.3f}",
            })
        pd.DataFrame(panel_c).to_csv(f, sep='\t', index=False)

    print(f"  Saved: TableS12_regression.tsv")


def create_table_s13():
    print("=== Creating Table S13: ChromHMM ===")

    chromhmm = pd.read_csv(os.path.join(ADD_DIR, 'prompt9_chromhmm_enrichment.tsv'), sep='\t')

    dnase = chromhmm[chromhmm['Modality'] == 'dnase'].copy()

    total_in = dnase['N_high_in'].values + dnase['N_low_in'].values
    total_out = dnase['N_high_out'].values + dnase['N_low_out'].values
    total_all = total_in + total_out

    rows = []
    for _, r in dnase.iterrows():
        total = r['N_high_in'] + r['N_low_in'] + r['N_high_out'] + r['N_low_out']
        pct_all = (r['N_high_in'] + r['N_low_in']) / total * 100
        rows.append({
            'Chromatin State': r['State_category'],
            '% All Variants': f"{pct_all:.1f}%",
            '% High-effect': f"{r['Pct_high']:.1f}%",
            '% Low-effect': f"{r['Pct_low']:.1f}%",
            'OR': f"{r['OR']:.2f}",
            'P-value': f"{r['Fisher_P']:.2e}" if r['Fisher_P'] < 0.01 else f"{r['Fisher_P']:.3f}",
        })

    df_out = pd.DataFrame(rows)

    with open(os.path.join(OUT_DIR, 'TableS13_chromhmm.tsv'), 'w') as f:
        f.write("# Supplementary Table S13. ChromHMM Regulatory Element Enrichment\n")
        f.write("# Roadmap E073 (dorsolateral prefrontal cortex), 15-state model\n")
        f.write("# High-effect = top 20% by DNase score\n\n")
        df_out.to_csv(f, sep='\t', index=False)

    print(f"  Saved: TableS13_chromhmm.tsv")


def create_table_s14():
    print("=== Creating Table S14: Decile ===")

    decile = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_decile_CE.tsv'), sep='\t')

    rna = decile[decile['Modality'] == 'rna_seq'].sort_values('Decile')
    chip = decile[decile['Modality'] == 'chip_histone'].sort_values('Decile')

    rows = []
    for i in range(len(rna)):
        rr = rna.iloc[i]
        cr = chip.iloc[i] if i < len(chip) else None
        row = {
            'Decile': f"D{int(rr['Decile'])}",
            'N variants (RNA-seq)': int(rr['N']),
            'RNA-seq Mean Score': f"{rr['Mean_effect']:.3f}",
            'RNA-seq CE%': f"{rr['CE_pct']:.1f}%",
        }
        if cr is not None:
            row['ChIP-histone CE%'] = f"{cr['CE_pct']:.1f}%"
            row['ChIP-histone Mean Score'] = f"{cr['Mean_effect']:.1f}"
        rows.append(row)

    df_out = pd.DataFrame(rows)

    rna_r, rna_p = stats.spearmanr(rna['Decile'], rna['CE_pct'])
    chip_r, chip_p = stats.spearmanr(chip['Decile'], chip['CE_pct'])

    with open(os.path.join(OUT_DIR, 'TableS14_decile.tsv'), 'w') as f:
        f.write("# Supplementary Table S14. Decile Case-Enrichment (Dose-Response)\n\n")
        df_out.to_csv(f, sep='\t', index=False)
        f.write(f"\n# RNA-seq: Spearman r = {rna_r:.3f}, P = {rna_p:.3f}\n")
        f.write(f"# ChIP-histone: Spearman r = {chip_r:.3f}, P = {chip_p:.3f}\n")

    print(f"  Saved: TableS14_decile.tsv")
    print(f"    RNA-seq dose-response: r={rna_r:.3f}, P={rna_p:.3f}")
    print(f"    ChIP-histone dose-response: r={chip_r:.3f}, P={chip_p:.3f}")


def create_figure5():
    print("=== Creating Figure 5: R5 Replication ===")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(mm2in(170), mm2in(80)),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    boot_all = pd.read_csv(os.path.join(R5_DIR, 'Phase_C_bootstrap_IR_all.tsv'), sep='\t')
    summary = pd.read_csv(os.path.join(R5_DIR, 'Phase_C_bootstrap_IR_summary.tsv'), sep='\t')

    modalities = ['RNA-seq', 'CAGE', 'DNase', 'ChIP-histone']
    mod_cols = ['RNA-seq_IR', 'CAGE_IR', 'DNase_IR', 'ChIP-histone_IR']
    colors = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_PURPLE]

    violin_data = []
    for col in mod_cols:
        violin_data.append(boot_all[col].values)

    parts = ax1.violinplot(violin_data, positions=range(len(modalities)),
                           showmeans=False, showmedians=True, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    r4_irs = [1.086, 1.056, 1.028, 1.062]
    r4_in_ci = []
    for i, mod in enumerate(modalities):
        s = summary[summary['Modality'] == mod].iloc[0]
        in_ci = s['R4_in_CI']
        r4_in_ci.append(in_ci)

        ax1.scatter(i, r4_irs[i], marker='^', color=CB_RED, s=60, zorder=5,
                   edgecolors='black', linewidths=0.5)

        if in_ci:
            ax1.text(i, s['R5_CI_high'] + 0.01, '*', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=CB_RED)

    ax1.axhline(y=1.0, color=CB_GRAY, linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_xticks(range(len(modalities)))
    ax1.set_xticklabels(modalities, rotation=20, ha='right')
    ax1.set_ylabel('Interaction Ratio (IR)')
    ax1.set_title('(a) R5 Bootstrap IR Distribution', fontweight='bold', fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=CB_RED,
               markeredgecolor='black', markersize=8, label='R4 Discovery IR'),
        Line2D([0], [0], color='black', linewidth=1.5, label='R5 Median'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True,
              framealpha=0.9, fontsize=6)

    by_pop = pd.read_csv(os.path.join(R5_DIR, 'Phase_B_by_population.tsv'), sep='\t')
    boot_pop = pd.read_csv(os.path.join(R5_DIR, 'Phase_C_bootstrap_IR_by_population.tsv'), sep='\t')

    pops = ['NHW', 'Hispanic', 'AA']
    x = np.arange(len(pops))
    width = 0.35

    unmatched_irs = []
    matched_irs = []
    matched_ci_lo = []
    matched_ci_hi = []

    for pop in pops:
        row = by_pop[by_pop['Population'] == pop].iloc[0]
        unmatched_irs.append(row['RNA-seq_IR'])

        bp = boot_pop[boot_pop['Population'] == pop]
        if len(bp) > 0:
            rna_vals = bp['RNA-seq_IR'].dropna()
            matched_irs.append(rna_vals.median())
            matched_ci_lo.append(rna_vals.quantile(0.025))
            matched_ci_hi.append(rna_vals.quantile(0.975))
        else:
            matched_irs.append(np.nan)
            matched_ci_lo.append(np.nan)
            matched_ci_hi.append(np.nan)

    matched_irs = np.array(matched_irs)
    matched_ci_lo = np.array(matched_ci_lo)
    matched_ci_hi = np.array(matched_ci_hi)

    bars1 = ax2.bar(x - width/2, unmatched_irs, width, label='Unmatched',
                     color=[CB_BLUE, CB_ORANGE, CB_GREEN], alpha=0.4,
                     edgecolor='gray', linewidth=0.5)

    bars2 = ax2.bar(x + width/2, matched_irs, width, label='Matched (Bootstrap)',
                     color=[CB_BLUE, CB_ORANGE, CB_GREEN], alpha=0.9,
                     edgecolor='black', linewidth=0.5)

    for i in range(len(pops)):
        if not np.isnan(matched_ci_lo[i]):
            ax2.errorbar(x[i] + width/2, matched_irs[i],
                        yerr=[[matched_irs[i] - matched_ci_lo[i]],
                              [matched_ci_hi[i] - matched_irs[i]]],
                        fmt='none', ecolor='black', capsize=3, linewidth=1)

    ax2.axhline(y=1.086, color=CB_RED, linestyle='--', linewidth=1, alpha=0.8,
               label='R4 IR (1.086)')
    ax2.axhline(y=1.0, color=CB_GRAY, linestyle='--', linewidth=0.8, alpha=0.5)

    ax2.annotate('n=49 AD', xy=(2 + width/2, matched_irs[2] + 0.02),
                fontsize=6, ha='center', style='italic')

    ax2.set_xticks(x)
    ax2.set_xticklabels(pops)
    ax2.set_ylabel('RNA-seq IR')
    ax2.set_title('(b) Per-Population IR (R5)', fontweight='bold', fontsize=9)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=6)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'Figure5_R5_replication.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: Figure5_R5_replication.pdf/png")


def create_figure_s8():
    print("=== Creating Figure S8: Gene IR Forest Plot ===")

    r4_boot = pd.read_csv(os.path.join(ADD_DIR, 'prompt6_gene_IR_bootstrap_R4.tsv'), sep='\t')
    rna = r4_boot[r4_boot['modality'] == 'rna_seq'].copy()
    rna = rna.sort_values('IR_original', ascending=True)

    n_genes = len(rna)
    fig, ax = plt.subplots(figsize=(mm2in(170), mm2in(max(250, n_genes * 5))))

    y_pos = range(n_genes)

    bold_positive = {'TNIP1', 'SORL1', 'SIGLEC11'}
    bold_negative = {'APH1B', 'CASP7', 'CD2AP'}

    for i, (_, r) in enumerate(rna.iterrows()):
        gene = r['gene']
        ir = r['IR_original']
        ci_lo = r['CI_lower']
        ci_hi = r['CI_upper']
        includes_1 = r['includes_1']

        if not includes_1 and ir > 1:
            color = CB_BLUE
        elif not includes_1 and ir < 1:
            color = CB_RED
        else:
            color = CB_GRAY

        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=1.2, zorder=2)
        ax.plot(ir, i, 'o', color=color, markersize=4, zorder=3)

        weight = 'bold' if gene in bold_positive or gene in bold_negative else 'normal'
        ax.text(-0.05, i, gene, ha='right', va='center', fontsize=6,
               fontweight=weight, transform=ax.get_yaxis_transform())

    ax.axvline(x=1.0, color=CB_GRAY, linestyle='--', linewidth=1, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([''] * n_genes)
    ax.set_xlabel('Interaction Ratio (RNA-seq)')
    ax.set_title('Supplementary Figure S8. Gene-Specific IR with Bootstrap 95% CI (R4)',
                fontweight='bold', fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=CB_BLUE, label='IR > 1 (CI excl. 1)', markersize=5, linewidth=1),
        Line2D([0], [0], marker='o', color=CB_RED, label='IR < 1 (CI excl. 1)', markersize=5, linewidth=1),
        Line2D([0], [0], marker='o', color=CB_GRAY, label='CI includes 1', markersize=5, linewidth=1),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=6)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'FigureS8_gene_IR_forest.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: FigureS8_gene_IR_forest.pdf/png ({n_genes} genes)")


def create_figure_s9():
    print("=== Creating Figure S9: R4 vs R5 Scatter ===")

    r4r5 = pd.read_csv(os.path.join(ADD_DIR, 'prompt6_R4_vs_R5_gene_IR.tsv'), sep='\t')

    rna = r4r5[r4r5['modality'] == 'rna_seq'].dropna(subset=['R4_IR', 'R5_IR']).copy()

    fig, ax = plt.subplots(figsize=(mm2in(120), mm2in(120)))

    ax.scatter(rna['R4_IR'], rna['R5_IR'], color=CB_BLUE, s=30, alpha=0.7,
              edgecolors='black', linewidths=0.3, zorder=3)

    for _, r in rna.iterrows():
        if pd.notna(r.get('R4_CI_lower')) and pd.notna(r.get('R5_CI_lower')):
            ax.errorbar(r['R4_IR'], r['R5_IR'],
                       xerr=[[r['R4_IR'] - r['R4_CI_lower']], [r['R4_CI_upper'] - r['R4_IR']]],
                       yerr=[[r['R5_IR'] - r['R5_CI_lower']], [r['R5_CI_upper'] - r['R5_IR']]],
                       fmt='none', ecolor=CB_GRAY, alpha=0.3, linewidth=0.5, zorder=2)

    lim_min = min(rna['R4_IR'].min(), rna['R5_IR'].min()) - 0.1
    lim_max = max(rna['R4_IR'].max(), rna['R5_IR'].max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color=CB_GRAY, linewidth=0.8, zorder=1)

    ax.axhline(y=1.0, color=CB_GRAY, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=1.0, color=CB_GRAY, linestyle=':', linewidth=0.5, alpha=0.5)

    label_genes = ['TNIP1', 'SORL1', 'APH1B', 'CD2AP', 'SIGLEC11', 'CASP7']
    for _, r in rna.iterrows():
        if r['gene'] in label_genes:
            offset = (5, 5) if r['R4_IR'] < 1.5 else (-5, -10)
            ax.annotate(r['gene'], (r['R4_IR'], r['R5_IR']),
                       xytext=offset, textcoords='offset points',
                       fontsize=6, fontweight='bold', zorder=4)

    sp_r, sp_p = stats.spearmanr(rna['R4_IR'], rna['R5_IR'])
    ax.text(0.05, 0.95, f'Spearman r = {sp_r:.3f}\nP = {sp_p:.3f}',
           transform=ax.transAxes, fontsize=7, va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('R4 Gene IR (RNA-seq)')
    ax.set_ylabel('R5 Gene IR (RNA-seq)')
    ax.set_title('Supplementary Figure S9. R4 vs R5 Gene-Specific IR',
                fontweight='bold', fontsize=9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'FigureS9_R4_vs_R5_geneIR.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: FigureS9_R4_vs_R5_geneIR.pdf/png (r={sp_r:.3f}, P={sp_p:.3f})")


def create_figure_s10():
    print("=== Creating Figure S10: Sensitivity ===")

    ac_df = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_sensitivity_AC.tsv'), sep='\t')
    pct_df = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_sensitivity_percentile.tsv'), sep='\t')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(mm2in(170), mm2in(80)))

    for mod, color, label in [('rna_seq', CB_BLUE, 'RNA-seq'), ('chip_histone', CB_ORANGE, 'ChIP-histone')]:
        d = ac_df[ac_df['Modality'] == mod]
        ax1.plot(d['AC_threshold'], d['IR'], '-o', color=color, label=label,
                markersize=5, linewidth=1.5)

    ax1.axhline(y=1.0, color=CB_GRAY, linestyle='--', linewidth=0.8)
    ax1.axvline(x=3, color=CB_GRAY, linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.text(3.2, ax1.get_ylim()[1] * 0.99, 'primary', fontsize=6, color=CB_GRAY, va='top')
    ax1.set_xlabel('AC Threshold')
    ax1.set_ylabel('Interaction Ratio')
    ax1.set_title('(a) AC Threshold Sensitivity', fontweight='bold')
    ax1.legend(fontsize=6, frameon=True)
    ax1.set_xticks([1, 3, 5, 10])

    for mod, color, label in [('rna_seq', CB_BLUE, 'RNA-seq'), ('chip_histone', CB_ORANGE, 'ChIP-histone')]:
        d = pct_df[pct_df['Modality'] == mod].drop_duplicates('Threshold_percentile').sort_values('Threshold_percentile')
        top_pct = 100 - d['Threshold_percentile'].values

        sig = d['P_value'] < 0.05
        ax2.plot(top_pct, d['IR'], '-', color=color, linewidth=1.5, label=label)
        ax2.scatter(top_pct[sig], d['IR'][sig], color=color, s=30, zorder=3, marker='o')
        ax2.scatter(top_pct[~sig], d['IR'][~sig], color=color, s=30, zorder=3,
                   marker='o', facecolors='white', edgecolors=color, linewidths=1)

    ax2.axhline(y=1.0, color=CB_GRAY, linestyle='--', linewidth=0.8)
    ax2.axvline(x=20, color=CB_GRAY, linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.text(20.5, ax2.get_ylim()[1] * 0.99, 'primary', fontsize=6, color=CB_GRAY, va='top')
    ax2.set_xlabel('Top Percentile (%)')
    ax2.set_ylabel('Interaction Ratio')
    ax2.set_title('(b) Percentile Threshold Sensitivity', fontweight='bold')
    ax2.legend(fontsize=6, frameon=True)
    ax2.invert_xaxis()

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'FigureS10_sensitivity.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: FigureS10_sensitivity.pdf/png")


def create_figure_s11():
    print("=== Creating Figure S11: Decile Dose-Response ===")

    decile = pd.read_csv(os.path.join(ADD_DIR, 'prompt8_decile_CE.tsv'), sep='\t')

    fig, ax = plt.subplots(figsize=(mm2in(120), mm2in(80)))

    for mod, color, label in [('rna_seq', CB_BLUE, 'RNA-seq'), ('chip_histone', CB_ORANGE, 'ChIP-histone')]:
        d = decile[decile['Modality'] == mod].sort_values('Decile')

        ax.plot(d['Decile'], d['CE_pct'], '-o', color=color, label=label,
               markersize=5, linewidth=1.5)

        slope, intercept, _, _, _ = stats.linregress(d['Decile'], d['CE_pct'])
        x_fit = np.linspace(1, 10, 50)
        ax.plot(x_fit, slope * x_fit + intercept, '--', color=color, linewidth=0.8, alpha=0.5)

        sp_r, sp_p = stats.spearmanr(d['Decile'], d['CE_pct'])
        label_text = f'{label}: r={sp_r:.3f}, P={sp_p:.3f}'

    rna_d = decile[decile['Modality'] == 'rna_seq'].sort_values('Decile')
    chip_d = decile[decile['Modality'] == 'chip_histone'].sort_values('Decile')
    rna_r, rna_p = stats.spearmanr(rna_d['Decile'], rna_d['CE_pct'])
    chip_r, chip_p = stats.spearmanr(chip_d['Decile'], chip_d['CE_pct'])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=CB_BLUE, label=f'RNA-seq (r={rna_r:.3f}, P={rna_p:.3f})',
               markersize=5, linewidth=1.5),
        Line2D([0], [0], marker='o', color=CB_ORANGE, label=f'ChIP-histone (r={chip_r:.3f}, P={chip_p:.3f})',
               markersize=5, linewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=6)

    ax.set_xlabel('AlphaGenome Score Decile')
    ax.set_ylabel('Case-Enrichment (%)')
    ax.set_title('Supplementary Figure S11. Decile Dose-Response',
                fontweight='bold', fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f'D{i}' for i in range(1, 11)])

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'FigureS11_decile_dose_response.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: FigureS11_decile_dose_response.pdf/png")


def create_figure_s12():
    print("=== Creating Figure S12: ChromHMM ===")

    chromhmm = pd.read_csv(os.path.join(ADD_DIR, 'prompt9_chromhmm_enrichment.tsv'), sep='\t')

    dnase = chromhmm[chromhmm['Modality'] == 'dnase'].copy()

    states = ['Active_Promoter', 'Enhancer', 'Transcription', 'Repressed', 'Quiescent']
    dnase = dnase[dnase['State_category'].isin(states)].copy()

    dnase['log_OR'] = np.log(dnase['OR'])
    dnase['SE_logOR'] = np.sqrt(
        1/dnase['N_high_in'].clip(lower=0.5) +
        1/dnase['N_high_out'].clip(lower=0.5) +
        1/dnase['N_low_in'].clip(lower=0.5) +
        1/dnase['N_low_out'].clip(lower=0.5)
    )
    dnase['OR_CI_lo'] = np.exp(dnase['log_OR'] - 1.96 * dnase['SE_logOR'])
    dnase['OR_CI_hi'] = np.exp(dnase['log_OR'] + 1.96 * dnase['SE_logOR'])

    state_order = ['Enhancer', 'Active_Promoter', 'Transcription', 'Repressed', 'Quiescent']
    dnase['State_category'] = pd.Categorical(dnase['State_category'], categories=state_order, ordered=True)
    dnase = dnase.sort_values('State_category')

    fig, ax = plt.subplots(figsize=(mm2in(120), mm2in(80)))

    y_pos = range(len(dnase))

    for i, (_, r) in enumerate(dnase.iterrows()):
        state = r['State_category']
        or_val = r['OR']
        ci_lo = r['OR_CI_lo']
        ci_hi = r['OR_CI_hi']

        if or_val > 1 and r['Fisher_P'] < 0.05:
            color = CB_RED
        elif or_val < 1 and r['Fisher_P'] < 0.05:
            color = CB_BLUE
        else:
            color = CB_GRAY

        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=2, zorder=2)
        ax.plot(or_val, i, 'o', color=color, markersize=7, zorder=3)

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace('_', ' ') for s in state_order])
    ax.set_xlabel('Odds Ratio (log scale)')
    ax.set_xscale('log')
    ax.set_title('Supplementary Figure S12. ChromHMM Enrichment\n(DNase High vs Low Effect)',
                fontweight='bold', fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=CB_RED, label='Enriched (P < 0.05)',
               markersize=6, linewidth=2),
        Line2D([0], [0], marker='o', color=CB_BLUE, label='Depleted (P < 0.05)',
               markersize=6, linewidth=2),
        Line2D([0], [0], marker='o', color=CB_GRAY, label='Not significant',
               markersize=6, linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=6)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'FigureS12_chromhmm.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: FigureS12_chromhmm.pdf/png")


def create_figure_s13():
    print("=== Creating Figure S13: Regression Forest Plot ===")

    firth = pd.read_csv(os.path.join(ADD_DIR, 'prompt7_firth_R4.tsv'), sep='\t')

    r4 = firth[(firth['Cohort'] == 'R4') & (firth['Model'] == 'single')]
    r5 = firth[(firth['Cohort'] == 'R5') & (firth['Model'] == 'single')]

    modalities = ['rna_seq', 'cage', 'dnase', 'chip_histone']
    mod_labels = ['RNA-seq', 'CAGE', 'DNase', 'ChIP-histone']

    fig, ax = plt.subplots(figsize=(mm2in(120), mm2in(80)))

    offset = 0.15

    for i, mod in enumerate(modalities):
        r4_row = r4[r4['Modality'] == mod].iloc[0]
        r4_or = r4_row['OR']
        r4_lo = r4_row['OR_CI_lower']
        r4_hi = r4_row['OR_CI_upper']
        r4_sig = r4_row['P_value'] < 0.05

        marker_r4 = 'o' if r4_sig else 'o'
        facecolor_r4 = CB_BLUE if r4_sig else 'white'

        ax.plot([r4_lo, r4_hi], [i + offset, i + offset], color=CB_BLUE, linewidth=1.5, zorder=2)
        ax.plot(r4_or, i + offset, 'o', color=CB_BLUE, markersize=7, zorder=3,
               markerfacecolor=facecolor_r4, markeredgecolor=CB_BLUE, markeredgewidth=1)

        r5_row = r5[r5['Modality'] == mod].iloc[0]
        r5_or = r5_row['OR']
        r5_lo = r5_row['OR_CI_lower']
        r5_hi = r5_row['OR_CI_upper']
        r5_sig = r5_row['P_value'] < 0.05

        facecolor_r5 = CB_RED if r5_sig else 'white'

        ax.plot([r5_lo, r5_hi], [i - offset, i - offset], color=CB_RED, linewidth=1.5, zorder=2)
        ax.plot(r5_or, i - offset, 'o', color=CB_RED, markersize=7, zorder=3,
               markerfacecolor=facecolor_r5, markeredgecolor=CB_RED, markeredgewidth=1)

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(mod_labels)
    ax.set_xlabel('Odds Ratio')
    ax.set_title('Supplementary Figure S13. Logistic Regression OR\n(R4 Discovery vs R5 Replication)',
                fontweight='bold', fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=CB_BLUE, markerfacecolor=CB_BLUE,
               label='R4 (P < 0.05)', markersize=7, linewidth=1.5),
        Line2D([0], [0], marker='o', color=CB_BLUE, markerfacecolor='white',
               markeredgecolor=CB_BLUE, label='R4 (n.s.)', markersize=7, linewidth=1.5),
        Line2D([0], [0], marker='o', color=CB_RED, markerfacecolor=CB_RED,
               label='R5 (P < 0.05)', markersize=7, linewidth=1.5),
        Line2D([0], [0], marker='o', color=CB_RED, markerfacecolor='white',
               markeredgecolor=CB_RED, label='R5 (n.s.)', markersize=7, linewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=6)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'FigureS13_regression_forest.{ext}'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: FigureS13_regression_forest.pdf/png")


def create_figure1_guide():
    print("=== Creating Figure 1 Revision Guide ===")

    guide = """# Figure 1 Revision Guide

- Shows R4 Discovery workflow only
- Pipeline: Gene Selection → Rare Variant Extraction → AlphaGenome Scoring →
  PLINK Frequency Analysis → Case-Control Ratio Analysis


- Add "Discovery Cohort (ADSP R4)" banner above existing diagram
- Add sample size: "N = 24,595 (AD 6,296, CN 18,299)"

- Position: Right side or below the Discovery workflow
- Box content:
  ┌─────────────────────────────────────────┐
  │  Independent Replication (ADSP R5-only) │
  │  N = 11,545 (AD 1,408, CN 10,137)      │
  │  Populations: NHW + Hispanic + AA       │
  │  Bootstrap 1:2.9 matching × 100         │
  └─────────────────────────────────────────┘

- Label: "Validation"
- Style: Thick arrow, possibly dashed

  ┌─────────────────────────────────────────────────┐
  │  Replication Results                             │
  │  • RNA-seq IR = 1.129 [1.075-1.201]  ✓ R4 in CI│
  │  • ChIP-histone IR = 1.107 [1.058-1.153]  ✓    │
  │  • CAGE IR = 1.151 [1.097-1.214]  (direction ✓) │
  │  • All 4 modalities IR > 1                       │
  └─────────────────────────────────────────────────┘

- Discovery boxes: Original colors (unchanged)
- Replication boxes: Slightly different shade or border style
  to visually distinguish Discovery vs Replication

- Small note/arrow to side:
  "Asian extension (N=1,035): RNA-seq IR = 1.533, P = 2.25×10⁻⁴"

- Option A: Side-by-side (Discovery left, Replication right)
  - Pro: Compact, shows parallel structure
  - Con: May be too wide for 1-column

- Option B: Stacked (Discovery top, Replication bottom)
  - Pro: Fits 1-column width (88mm)
  - Con: Taller figure

- Recommendation: Option B (stacked) for Nature Communications 1-column format
"""

    with open(os.path.join(OUT_DIR, 'Figure1_revision_guide.txt'), 'w') as f:
        f.write(guide)

    print(f"  Saved: Figure1_revision_guide.txt")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Manuscript Revision: Tables and Figures Generation")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print()

    print("--- PART 1: Main Text Tables ---")
    create_table6()
    create_table7()
    print()

    print("--- PART 2: Supplementary Tables ---")
    create_table_s9()
    create_table_s10()
    create_table_s11()
    create_table_s12()
    create_table_s13()
    create_table_s14()
    print()

    print("--- PART 3: Main Figure ---")
    create_figure5()
    print()

    print("--- PART 4: Supplementary Figures ---")
    create_figure_s8()
    create_figure_s9()
    create_figure_s10()
    create_figure_s11()
    create_figure_s12()
    create_figure_s13()
    print()

    print("--- PART 5: Figure 1 Revision Guide ---")
    create_figure1_guide()
    print()

    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    for root, dirs, files in os.walk(OUT_DIR):
        for f in sorted(files):
            if f == 'create_tables_and_figures.py':
                continue
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, OUT_DIR):50s} {size:>10,d} bytes")


if __name__ == '__main__':
    main()
