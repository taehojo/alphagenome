#!/usr/bin/env python3
"""
Analysis C: Bulk RNA-seq Validation
====================================
Validates gene-level findings with independent bulk RNA-seq data
from GSE174367 (Morabito et al. 2021).

Input: GSE174367_bulkRNA_processed.rda.gz (R format, converted to CSV)
Output: Figures C1-C3 + statistical tables

Author: Taeho Jo (tjo@iu.edu)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import warnings
warnings.filterwarnings('ignore')

from utils_geo import (PROJECT_DIR, GEO_DIR, RESULTS_BASE, CELL_TYPE_GENES,
                        COLORS, GENE_LIST, get_gene_celltype_map, get_ad_genes,
                        load_variant_data, ensure_dir)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

OUTPUT_DIR = ensure_dir(f"{RESULTS_BASE}/C")
RDA_PATH = f"{GEO_DIR}/GSE174367_bulkRNA_processed.rda.gz"


def convert_rda_to_csv():
    """Convert R .rda/.rda.gz file to CSV using Rscript."""
    print("Converting R data file to CSV...")

    csv_dir = f"{GEO_DIR}/bulk_csv"
    os.makedirs(csv_dir, exist_ok=True)

    # Check if already converted
    existing_csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv')] if os.path.exists(csv_dir) else []
    if existing_csvs:
        print(f"  Already converted: {existing_csvs}")
        return csv_dir

    # Try decompressed .rda first, then .rda.gz
    rda_path = RDA_PATH.replace('.gz', '')
    if not os.path.exists(rda_path):
        if os.path.exists(RDA_PATH):
            print(f"  Decompressing {RDA_PATH}...")
            subprocess.run(['gunzip', '-k', RDA_PATH], check=True)
        else:
            raise FileNotFoundError(f"Bulk RNA-seq file not found: {RDA_PATH}")

    r_script = f"""
    load("{rda_path}")
    objs <- ls()
    cat("Objects in RDA file:\\n")
    for (n in objs) {{
        obj <- get(n)
        cat(sprintf("  %s: %s [%s]\\n", n, paste(class(obj), collapse=","), paste(dim(obj), collapse="x")))
        if (is.data.frame(obj) || is.matrix(obj)) {{
            write.csv(obj, file.path("{csv_dir}", paste0(n, ".csv")), row.names=TRUE)
            cat(sprintf("    -> Saved as %s.csv\\n", n))
        }}
    }}
    """

    r_script_path = f"{csv_dir}/convert.R"
    with open(r_script_path, 'w') as f:
        f.write(r_script)

    # Try module load r first
    result = subprocess.run(
        ['bash', '-c', f'source /etc/profile.d/modules.sh 2>/dev/null; module load r/4.4.1 2>/dev/null; Rscript {r_script_path}'],
        capture_output=True, text=True, timeout=300
    )
    print(result.stdout)
    if result.returncode != 0:
        # Try direct Rscript
        result2 = subprocess.run(['Rscript', r_script_path], capture_output=True, text=True, timeout=300)
        print(result2.stdout)
        if result2.returncode != 0:
            print(f"  R stderr: {result2.stderr}")
            raise RuntimeError("Failed to convert R file. Ensure R is available.")

    os.remove(r_script_path)
    return csv_dir


def load_bulk_rnaseq(csv_dir):
    """Load converted bulk RNA-seq data."""
    print("\nLoading bulk RNA-seq data...")

    # Known structure from GSE174367:
    # normExpr.reg.csv: expression matrix (ENSEMBL IDs x samples)
    # targets.csv: sample metadata (with Diagnosis column)
    expr_path = os.path.join(csv_dir, 'normExpr.reg.csv')
    targets_path = os.path.join(csv_dir, 'targets.csv')

    expr_df = None
    sample_info = None

    if os.path.exists(expr_path):
        expr_df = pd.read_csv(expr_path, index_col=0)
        print(f"  Expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    else:
        # Fallback: scan directory
        csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        print(f"  Available CSVs: {csvs}")
        for csv_file in csvs:
            path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(path, index_col=0)
            if df.shape[0] > 1000 and df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).mean() > 0.5:
                expr_df = df
                print(f"  Using {csv_file} as expression: {df.shape}")
                break

    if os.path.exists(targets_path):
        sample_info = pd.read_csv(targets_path, index_col=0)
        print(f"  Sample info: {sample_info.shape[0]} samples x {sample_info.shape[1]} cols")

    # Map ENSEMBL IDs to gene symbols using Table S1
    if expr_df is not None and expr_df.index[0].startswith('ENSG'):
        print("\n  Gene IDs are ENSEMBL format, mapping to symbols...")
        gene_list = pd.read_csv(GENE_LIST)
        # Build ENSEMBL -> symbol map
        ensg_to_symbol = dict(zip(gene_list['gene_id'], gene_list['gene_name']))

        # Also strip version numbers from expression index (ENSG00000142192.20 -> ENSG00000142192)
        expr_df.index = expr_df.index.astype(str).map(lambda x: x.split('.')[0])

        # Map what we can
        mapped = {ensg: sym for ensg, sym in ensg_to_symbol.items() if ensg in expr_df.index}
        print(f"  Mapped {len(mapped)}/{len(ensg_to_symbol)} AD genes to expression data")

        # Rename matched rows
        expr_df = expr_df.rename(index=mapped)

    if expr_df is not None:
        print(f"\n  Final expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
        print(f"  Gene examples: {list(expr_df.index[:5])}")

    return expr_df, sample_info


def identify_ad_control_samples(expr_df, sample_info):
    """Identify AD and Control samples from column names or sample info."""
    print("\nIdentifying AD vs Control samples...")

    ad_samples = []
    ctrl_samples = []

    if sample_info is not None:
        # The index of targets.csv is row number; SampleID or Sample.ID has sample names
        # Look for sample ID column that matches expression columns
        sample_id_col = None
        for col in ['SampleID', 'Sample.ID', 'Sample_ID']:
            if col in sample_info.columns:
                # Check if values match expression columns
                vals = set(sample_info[col].astype(str))
                overlap = vals & set(expr_df.columns)
                if len(overlap) > 0:
                    sample_id_col = col
                    break

        # If index matches expression columns directly
        if sample_id_col is None:
            idx_overlap = set(sample_info.index.astype(str)) & set(expr_df.columns)
            if len(idx_overlap) > 5:
                sample_info = sample_info.copy()
                sample_info['_sample_name'] = sample_info.index.astype(str)
                sample_id_col = '_sample_name'

        # Find diagnosis column
        diag_col = None
        for col in sample_info.columns:
            if col.lower() == 'diagnosis':
                diag_col = col
                break
        if diag_col is None:
            for col in sample_info.columns:
                col_lower = col.lower()
                if any(k in col_lower for k in ['diagnosis', 'disease', 'condition', 'status']):
                    diag_col = col
                    break

        if sample_id_col and diag_col:
            print(f"  Sample ID column: {sample_id_col}")
            print(f"  Diagnosis column: {diag_col}")
            print(f"  Diagnosis values: {sample_info[diag_col].unique()}")

            for _, row in sample_info.iterrows():
                sample_name = str(row[sample_id_col])
                diag = str(row[diag_col]).strip()
                if sample_name not in expr_df.columns:
                    continue
                if diag == 'AD':
                    ad_samples.append(sample_name)
                elif diag == 'Control':
                    ctrl_samples.append(sample_name)
        else:
            print(f"  WARNING: Could not find matching sample ID or diagnosis columns")

    # Fallback: parse column names
    if not ad_samples and not ctrl_samples:
        for col in expr_df.columns:
            col_lower = str(col).lower()
            if 'ad' in col_lower or 'alzheimer' in col_lower:
                ad_samples.append(col)
            elif 'ctrl' in col_lower or 'control' in col_lower:
                ctrl_samples.append(col)

    if not ad_samples and not ctrl_samples:
        print("  WARNING: Could not determine AD/Control groups.")
        return None, None

    print(f"  AD samples: {len(ad_samples)}")
    print(f"  Control samples: {len(ctrl_samples)}")

    return ad_samples, ctrl_samples


def analysis_bulk_de(expr_df, ad_samples, ctrl_samples, ad_genes):
    """Differential expression for AD genes."""
    print("\n" + "="*70)
    print("Bulk RNA-seq Differential Expression")
    print("="*70)

    if ad_samples is None or ctrl_samples is None:
        print("  No sample groups identified.")
        return None

    if len(ad_samples) < 2 or len(ctrl_samples) < 2:
        print(f"  Insufficient samples: {len(ad_samples)} AD, {len(ctrl_samples)} Control")
        return None

    # Find AD genes in expression data
    available_genes = set(expr_df.index.astype(str))
    found_genes = sorted(ad_genes & available_genes)
    print(f"  AD genes in bulk data: {len(found_genes)}/{len(ad_genes)}")

    if len(found_genes) == 0:
        # Try case-insensitive matching
        gene_map = {g.upper(): g for g in expr_df.index.astype(str)}
        found_genes = []
        for g in ad_genes:
            if g.upper() in gene_map:
                found_genes.append(gene_map[g.upper()])
        print(f"  After case-insensitive matching: {len(found_genes)}")

    results = []
    gene_ct = get_gene_celltype_map()

    for gene in found_genes:
        ad_vals = expr_df.loc[gene, ad_samples].values.astype(float)
        ctrl_vals = expr_df.loc[gene, ctrl_samples].values.astype(float)

        ad_mean = np.mean(ad_vals)
        ctrl_mean = np.mean(ctrl_vals)
        log2fc = np.log2((ad_mean + 1e-10) / (ctrl_mean + 1e-10))

        stat, pval = stats.mannwhitneyu(ad_vals, ctrl_vals, alternative='two-sided')

        results.append({
            'gene': gene,
            'cell_type': gene_ct.get(gene, 'Ubiquitous'),
            'ad_mean': ad_mean,
            'ctrl_mean': ctrl_mean,
            'log2fc': log2fc,
            'mannwhitney_U': stat,
            'pvalue': pval,
        })

    if not results:
        return None

    de_df = pd.DataFrame(results)
    de_df['fdr'] = multipletests(de_df['pvalue'], method='fdr_bh')[1]

    sig = de_df[de_df['fdr'] < 0.05]
    print(f"\n  Total genes tested: {len(de_df)}")
    print(f"  Significant (FDR<0.05): {len(sig)}")

    # Top genes
    top = de_df.nsmallest(10, 'pvalue')
    print("\n  Top 10 genes:")
    for _, row in top.iterrows():
        direction = "UP" if row['log2fc'] > 0 else "DOWN"
        print(f"    {row['gene']}: log2FC={row['log2fc']:.3f} ({direction}), P={row['pvalue']:.2e}, FDR={row['fdr']:.2e}")

    return de_df


def analysis_expression_profiling(expr_df, ad_genes):
    """Expression profiling (no AD/Control comparison needed)."""
    print("\n" + "="*70)
    print("Bulk RNA-seq Expression Profiling (all samples)")
    print("="*70)

    available_genes = set(expr_df.index.astype(str))
    found_genes = sorted(ad_genes & available_genes)

    if not found_genes:
        gene_map = {g.upper(): g for g in expr_df.index.astype(str)}
        found_genes = [gene_map[g.upper()] for g in ad_genes if g.upper() in gene_map]

    print(f"  AD genes found: {len(found_genes)}")

    if not found_genes:
        return None

    gene_ct = get_gene_celltype_map()
    profile = []
    for gene in found_genes:
        vals = expr_df.loc[gene].values.astype(float)
        profile.append({
            'gene': gene,
            'cell_type': gene_ct.get(gene, 'Ubiquitous'),
            'mean_expr': np.mean(vals),
            'std_expr': np.std(vals),
            'median_expr': np.median(vals),
        })

    return pd.DataFrame(profile)


def analysis_ccratio_vs_bulk(de_df, variant_df):
    """Correlate cc_ratio with bulk expression change."""
    print("\n" + "="*70)
    print("CC Ratio vs Bulk Expression Change")
    print("="*70)

    if de_df is None:
        print("  No DE data.")
        return None

    gene_cc = variant_df.groupby('gene_name').agg({
        'cc_ratio': 'mean',
        'cell_type': 'first'
    }).reset_index()
    gene_cc.columns = ['gene', 'mean_cc_ratio', 'variant_celltype']

    merged = de_df[['gene', 'log2fc', 'cell_type']].merge(gene_cc, on='gene')

    if len(merged) < 5:
        print(f"  Too few overlapping genes (n={len(merged)})")
        return None

    r, p = stats.spearmanr(merged['mean_cc_ratio'], merged['log2fc'])
    print(f"  Overall: Spearman r={r:.3f}, P={p:.2e}, n={len(merged)}")

    # By cell type
    for ct in ['Neuron', 'Microglia', 'Astrocyte']:
        ct_data = merged[merged['cell_type'] == ct]
        if len(ct_data) >= 3:
            r_ct, p_ct = stats.spearmanr(ct_data['mean_cc_ratio'], ct_data['log2fc'])
            print(f"  {ct}: r={r_ct:.3f}, P={p_ct:.2e}, n={len(ct_data)}")

    return merged


# ============================================================
# Figure generation
# ============================================================

def figure_C1_volcano(de_df):
    """Figure C1: DE volcano plot."""
    print("\nGenerating Figure C1: Volcano plot...")

    if de_df is None or len(de_df) == 0:
        print("  No DE data!")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    de_df = de_df.copy()
    de_df['neg_log10_fdr'] = -np.log10(de_df['fdr'].clip(lower=1e-50))
    de_df['significant'] = de_df['fdr'] < 0.05

    # Plot by cell type
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        ct_data = de_df[de_df['cell_type'] == ct]
        ns = ct_data[~ct_data['significant']]
        sig = ct_data[ct_data['significant']]

        ax.scatter(ns['log2fc'], ns['neg_log10_fdr'],
                   c=COLORS[ct], s=20, alpha=0.3, label=None)
        ax.scatter(sig['log2fc'], sig['neg_log10_fdr'],
                   c=COLORS[ct], s=35, alpha=0.8, edgecolor='white', linewidth=0.3,
                   label=f'{ct} (sig: {len(sig)})')

    # Label top genes
    top = de_df.nlargest(8, 'neg_log10_fdr')
    for _, row in top.iterrows():
        ax.annotate(row['gene'], (row['log2fc'], row['neg_log10_fdr']),
                    fontsize=7, fontweight='bold', color=COLORS.get(row['cell_type'], '#333'),
                    xytext=(5, 5), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', color='#999', lw=0.4, shrinkA=0, shrinkB=2))

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('log2 Fold Change (AD/Control)')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title('Bulk RNA-seq Differential Expression', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_C1_volcano.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def figure_C2_ccratio_vs_bulk(merged_df):
    """Figure C2: cc_ratio vs bulk expression change."""
    print("\nGenerating Figure C2: CC ratio vs bulk expression...")

    if merged_df is None or len(merged_df) < 3:
        print("  Insufficient data!")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        ct_data = merged_df[merged_df['cell_type'] == ct]
        if len(ct_data) > 0:
            ax.scatter(ct_data['mean_cc_ratio'], ct_data['log2fc'],
                       c=COLORS[ct], s=30, alpha=0.7, edgecolor='white', linewidth=0.3,
                       label=f'{ct} (n={len(ct_data)})')

    # Overall correlation
    r, p = stats.spearmanr(merged_df['mean_cc_ratio'], merged_df['log2fc'])

    # Label extremes
    for col, n in [('mean_cc_ratio', 3), ('log2fc', 3)]:
        extremes = pd.concat([merged_df.nlargest(n, col), merged_df.nsmallest(n, col)]).drop_duplicates('gene')
        for _, row in extremes.iterrows():
            ax.annotate(row['gene'], (row['mean_cc_ratio'], row['log2fc']),
                        fontsize=6, xytext=(4, 3), textcoords='offset points')

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Mean CC Ratio (case/control allele frequency)')
    ax.set_ylabel('log2 FC (AD/Control bulk expression)')
    ax.set_title(f'CC Ratio vs Bulk Expression\nSpearman r={r:.3f}, P={p:.2e}', fontweight='bold')
    ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_C2_ccratio_vs_bulk.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def figure_C3_celltype_foldchange(de_df):
    """Figure C3: Cell-type group fold-change comparison."""
    print("\nGenerating Figure C3: Cell-type fold-change...")

    if de_df is None or len(de_df) == 0:
        print("  No DE data!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Panel a: Mean log2FC by cell type
    ax = axes[0]
    cell_types = ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']
    means, sems, ns = [], [], []
    for ct in cell_types:
        ct_data = de_df[de_df['cell_type'] == ct]['log2fc']
        if len(ct_data) > 0:
            means.append(ct_data.mean())
            sems.append(ct_data.std() / np.sqrt(len(ct_data)))
            ns.append(len(ct_data))
        else:
            means.append(0)
            sems.append(0)
            ns.append(0)

    x = np.arange(len(cell_types))
    bars = ax.bar(x, means, yerr=sems, capsize=3,
                  color=[COLORS[ct] for ct in cell_types], alpha=0.85,
                  edgecolor='white', linewidth=0.5, width=0.65,
                  error_kw={'linewidth': 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([f'{ct}\n(n={n})' for ct, n in zip(cell_types, ns)], fontsize=7)
    ax.set_ylabel('Mean log2 FC (AD/Control)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('(a) Mean Fold-Change by Cell Type', fontweight='bold', loc='left')

    # Kruskal-Wallis test
    groups = [de_df[de_df['cell_type'] == ct]['log2fc'].values for ct in cell_types if
              len(de_df[de_df['cell_type'] == ct]) > 0]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        ax.text(0.95, 0.95, f'KW P={p_val:.2e}', transform=ax.transAxes,
                fontsize=7, ha='right', va='top')

    # Panel b: Fraction up/down by cell type
    ax = axes[1]
    for i, ct in enumerate(cell_types):
        ct_data = de_df[de_df['cell_type'] == ct]
        if len(ct_data) == 0:
            continue
        up = (ct_data['log2fc'] > 0).sum()
        down = (ct_data['log2fc'] < 0).sum()
        total = len(ct_data)
        ax.bar(i, up/total * 100, color=COLORS[ct], alpha=0.85, edgecolor='white')
        ax.bar(i, -down/total * 100, color=COLORS[ct], alpha=0.4, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, fontsize=7)
    ax.set_ylabel('% Genes')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('(b) Up vs Down Regulation', fontweight='bold', loc='left')
    ax.text(0.95, 0.95, 'Solid = Up, Light = Down', transform=ax.transAxes,
            fontsize=6, ha='right', va='top')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_C3_celltype_foldchange.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("="*70)
    print("Analysis C: Bulk RNA-seq Validation")
    print("="*70)

    ad_genes = get_ad_genes()
    variant_df = load_variant_data()

    # Convert R data
    try:
        csv_dir = convert_rda_to_csv()
    except Exception as e:
        print(f"\nERROR converting R file: {e}")
        print("Trying alternative approach...")

        # Check if .rda.gz exists
        if not os.path.exists(RDA_PATH):
            print(f"  File not found: {RDA_PATH}")
            print("  Please download the file first: bash download_GSE174367.sh")
            return

        # Try pyreadr if available
        try:
            import pyreadr
            result = pyreadr.read_r(RDA_PATH)
            csv_dir = f"{GEO_DIR}/bulk_csv"
            os.makedirs(csv_dir, exist_ok=True)
            for name, df in result.items():
                df.to_csv(f"{csv_dir}/{name}.csv")
                print(f"  Saved: {name}.csv ({df.shape})")
        except ImportError:
            print("  pyreadr not available. Install with: pip install pyreadr")
            print("  Or ensure R is available: module load r")
            return

    # Load data
    expr_df, sample_info = load_bulk_rnaseq(csv_dir)

    if expr_df is None:
        print("\nERROR: Could not load expression data!")
        return

    # Identify AD vs Control
    ad_samples, ctrl_samples = identify_ad_control_samples(expr_df, sample_info)

    # DE analysis
    de_df = None
    if ad_samples and ctrl_samples:
        de_df = analysis_bulk_de(expr_df, ad_samples, ctrl_samples, ad_genes)
    else:
        print("\n  No AD/Control sample groups found. Running profiling only.")

    # Expression profiling (always)
    profile_df = analysis_expression_profiling(expr_df, ad_genes)

    # CC ratio vs bulk expression
    merged_df = None
    if de_df is not None:
        merged_df = analysis_ccratio_vs_bulk(de_df, variant_df)

    # Generate figures
    if de_df is not None:
        figure_C1_volcano(de_df)
    if merged_df is not None:
        figure_C2_ccratio_vs_bulk(merged_df)
    if de_df is not None:
        figure_C3_celltype_foldchange(de_df)

    # Save tables
    if de_df is not None:
        de_df.to_csv(f"{OUTPUT_DIR}/bulk_DE_results.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/bulk_DE_results.csv")

    if profile_df is not None:
        profile_df.to_csv(f"{OUTPUT_DIR}/bulk_expression_profile.csv", index=False)
        print(f"Saved: {OUTPUT_DIR}/bulk_expression_profile.csv")

    if merged_df is not None:
        merged_df.to_csv(f"{OUTPUT_DIR}/ccratio_vs_bulk_expression.csv", index=False)
        print(f"Saved: {OUTPUT_DIR}/ccratio_vs_bulk_expression.csv")

    print("\n" + "="*70)
    print("Analysis C Complete")
    print("="*70)


if __name__ == '__main__':
    main()
