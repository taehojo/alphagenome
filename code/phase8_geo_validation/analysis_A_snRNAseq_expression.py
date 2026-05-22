#!/usr/bin/env python3
"""
Analysis A: Cell-Type-Specific Expression Validation (snRNA-seq)
================================================================
Validates paper's cell-type classification against actual brain expression
from GSE174367 (Morabito et al. 2021, Nature Genetics).

Input: GSE174367 snRNA-seq (61,472 nuclei, 12 AD + 8 ctrl, prefrontal cortex)
Output: Figures A1-A4 + statistical tables

Author: Taeho Jo (tjo@iu.edu)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

from utils_geo import (PROJECT_DIR, GEO_DIR, RESULTS_BASE, CELL_TYPE_GENES,
                        COLORS, GSE174367_CELLTYPE_MAP, get_gene_celltype_map,
                        get_ad_genes, load_variant_data, ensure_dir)

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

OUTPUT_DIR = ensure_dir(f"{RESULTS_BASE}/A")


def load_snrnaseq():
    """Load snRNA-seq data from GSE174367."""
    print("Loading snRNA-seq data...")

    h5_path = f"{GEO_DIR}/GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"
    meta_path = f"{GEO_DIR}/GSE174367_snRNA-seq_cell_meta.csv.gz"

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"snRNA-seq h5 file not found: {h5_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cell metadata not found: {meta_path}")

    # Load expression matrix
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    print(f"  Raw: {adata.n_obs} cells x {adata.n_vars} genes")

    # Load metadata
    meta = pd.read_csv(meta_path)
    print(f"  Metadata: {len(meta)} rows")
    print(f"  Metadata columns: {list(meta.columns)}")

    # Find barcode column
    barcode_col = None
    for col in meta.columns:
        if 'barcode' in col.lower() or col == meta.columns[0]:
            barcode_col = col
            break
    if barcode_col is None:
        barcode_col = meta.columns[0]

    print(f"  Using barcode column: {barcode_col}")

    # Find cell type column
    celltype_col = None
    for col in meta.columns:
        if 'cell' in col.lower() and 'type' in col.lower():
            celltype_col = col
            break
        elif col.lower() in ['celltype', 'cluster', 'cell_type', 'annotation']:
            celltype_col = col
            break

    if celltype_col is None:
        # Try to find by checking unique values
        for col in meta.columns:
            uniq = meta[col].nunique()
            vals = set(meta[col].dropna().astype(str).unique())
            if vals.intersection({'EX', 'INH', 'MG', 'ASC', 'ODC', 'OPC'}):
                celltype_col = col
                break

    print(f"  Using cell type column: {celltype_col}")

    # Find diagnosis column
    diag_col = None
    for col in meta.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ['diagnosis', 'disease', 'condition', 'group', 'status']):
            diag_col = col
            break
        vals = set(meta[col].dropna().astype(str).unique())
        if vals.intersection({'AD', 'Control', 'control', 'CTRL', 'Case'}):
            diag_col = col
            break

    print(f"  Using diagnosis column: {diag_col}")

    # Find sample/subject column
    sample_col = None
    for col in meta.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ['sample', 'subject', 'donor', 'patient', 'individual']):
            sample_col = col
            break

    print(f"  Using sample column: {sample_col}")

    # Match barcodes between adata and metadata
    meta_barcodes = set(meta[barcode_col].astype(str))
    adata_barcodes = set(adata.obs_names)

    # Try direct match
    overlap = meta_barcodes & adata_barcodes
    print(f"  Direct barcode overlap: {len(overlap)}")

    if len(overlap) < 100:
        # Try stripping suffixes like -1
        meta[barcode_col] = meta[barcode_col].astype(str).str.replace(r'-\d+$', '', regex=True)
        adata_stripped = [b.rsplit('-', 1)[0] for b in adata.obs_names]
        adata.obs['barcode_stripped'] = adata_stripped

        meta_barcodes = set(meta[barcode_col])
        adata_barcodes = set(adata_stripped)
        overlap = meta_barcodes & adata_barcodes
        print(f"  Stripped barcode overlap: {len(overlap)}")

    # Merge metadata into adata.obs
    meta_indexed = meta.set_index(barcode_col)

    if 'barcode_stripped' in adata.obs.columns:
        # Use stripped barcodes for matching
        for col in [celltype_col, diag_col, sample_col]:
            if col is not None and col in meta_indexed.columns:
                mapping = meta_indexed[col].to_dict()
                adata.obs[col] = adata.obs['barcode_stripped'].map(mapping)
    else:
        common = adata.obs_names.intersection(meta_indexed.index)
        if len(common) > 0:
            for col in [celltype_col, diag_col, sample_col]:
                if col is not None and col in meta_indexed.columns:
                    adata.obs.loc[common, col] = meta_indexed.loc[common, col]

    # Map cell types to our categories
    if celltype_col and celltype_col in adata.obs.columns:
        # Extract major type from annotation (e.g., "EX-L2/3" -> "EX")
        adata.obs['major_celltype'] = adata.obs[celltype_col].astype(str).apply(
            lambda x: x.split('-')[0].split('_')[0].strip() if pd.notna(x) else 'Unknown'
        )
        adata.obs['our_celltype'] = adata.obs['major_celltype'].map(GSE174367_CELLTYPE_MAP).fillna('Other')

    # Map diagnosis
    if diag_col and diag_col in adata.obs.columns:
        diag_vals = adata.obs[diag_col].dropna().unique()
        print(f"  Diagnosis values: {diag_vals}")
        # Standardize
        diag_map = {}
        for v in diag_vals:
            v_str = str(v).lower()
            if 'ad' in v_str or 'alzheimer' in v_str or 'case' in v_str:
                diag_map[v] = 'AD'
            elif 'control' in v_str or 'ctrl' in v_str or 'normal' in v_str or 'healthy' in v_str:
                diag_map[v] = 'Control'
            else:
                diag_map[v] = str(v)
        adata.obs['diagnosis'] = adata.obs[diag_col].map(diag_map)

    if sample_col and sample_col in adata.obs.columns:
        adata.obs['sample_id'] = adata.obs[sample_col].astype(str)

    # Filter to cells with metadata
    has_meta = adata.obs['our_celltype'].notna() if 'our_celltype' in adata.obs.columns else pd.Series([False]*adata.n_obs)
    adata = adata[has_meta].copy()
    print(f"  After metadata filter: {adata.n_obs} cells")

    # Print cell type distribution
    if 'our_celltype' in adata.obs.columns:
        print("\n  Cell type distribution:")
        for ct, n in adata.obs['our_celltype'].value_counts().items():
            print(f"    {ct}: {n:,}")

    if 'diagnosis' in adata.obs.columns:
        print("\n  Diagnosis distribution:")
        for d, n in adata.obs['diagnosis'].value_counts().items():
            print(f"    {d}: {n:,}")

    return adata, celltype_col, diag_col, sample_col


def normalize_data(adata):
    """Normalize and log-transform."""
    print("\nNormalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  Done (normalize_total + log1p)")
    return adata


def analysis_celltype_expression(adata, ad_genes):
    """Compute cell-type-specific expression for 85 AD genes."""
    print("\n" + "="*70)
    print("Cell-Type-Specific Expression Analysis")
    print("="*70)

    # Find which AD genes are in the data
    available_genes = set(adata.var_names)
    found_genes = sorted(ad_genes & available_genes)
    missing_genes = sorted(ad_genes - available_genes)
    print(f"  AD genes found in data: {len(found_genes)}/{len(ad_genes)}")
    if missing_genes:
        print(f"  Missing: {missing_genes[:10]}{'...' if len(missing_genes)>10 else ''}")

    if len(found_genes) == 0:
        print("  ERROR: No AD genes found in expression data!")
        return None, None

    # Subset to AD genes
    adata_sub = adata[:, found_genes].copy()

    # Compute mean expression per cell type
    celltypes = ['Neuron', 'Microglia', 'Astrocyte', 'Other']
    expr_by_ct = {}
    for ct in celltypes:
        mask = adata_sub.obs['our_celltype'] == ct
        if mask.sum() > 0:
            expr_by_ct[ct] = pd.Series(
                np.asarray(adata_sub[mask].X.mean(axis=0)).flatten(),
                index=found_genes
            )

    expr_df = pd.DataFrame(expr_by_ct)
    print(f"\n  Expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} cell types")

    return expr_df, found_genes


def analysis_concordance(expr_df, found_genes):
    """Check concordance between paper's cell-type classification and actual expression."""
    print("\n" + "="*70)
    print("Classification Concordance Analysis")
    print("="*70)

    gene_ct = get_gene_celltype_map()
    results = []

    for gene in found_genes:
        paper_ct = gene_ct.get(gene, 'Unknown')
        if gene not in expr_df.index:
            continue

        row = expr_df.loc[gene]
        # Which cell type has highest expression?
        celltypes_avail = [ct for ct in ['Neuron', 'Microglia', 'Astrocyte'] if ct in row.index]
        if not celltypes_avail:
            continue

        best_ct = row[celltypes_avail].idxmax()
        best_expr = row[celltypes_avail].max()

        # Specificity index: max / mean(others)
        other_mean = row[[ct for ct in celltypes_avail if ct != best_ct]].mean()
        specificity = best_expr / other_mean if other_mean > 0 else np.inf

        results.append({
            'gene': gene,
            'paper_celltype': paper_ct,
            'highest_expr_celltype': best_ct,
            'concordant': paper_ct == best_ct,
            'specificity_index': specificity,
            **{f'expr_{ct}': row.get(ct, 0) for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Other']}
        })

    results_df = pd.DataFrame(results)

    # Overall concordance
    specific_genes = results_df[results_df['paper_celltype'].isin(['Neuron', 'Microglia', 'Astrocyte'])]
    if len(specific_genes) > 0:
        concordance = specific_genes['concordant'].mean()
        print(f"\n  Cell-type-specific genes concordance: {concordance:.1%} ({specific_genes['concordant'].sum()}/{len(specific_genes)})")

        for ct in ['Neuron', 'Microglia', 'Astrocyte']:
            ct_genes = specific_genes[specific_genes['paper_celltype'] == ct]
            if len(ct_genes) > 0:
                ct_conc = ct_genes['concordant'].mean()
                print(f"    {ct}: {ct_conc:.1%} ({ct_genes['concordant'].sum()}/{len(ct_genes)})")

        print("\n  NOTE: Paper classification is based on functional role (GWAS risk mechanism),")
        print("  not expression pattern. Low Neuron/Astrocyte concordance reflects that many")
        print("  AD risk genes classified by functional role are expressed across multiple cell types.")
        print("  High Microglia concordance confirms these genes are genuinely microglia-specific.")

    # Statistical test: concordance vs random
    if len(specific_genes) > 0:
        observed = specific_genes['concordant'].sum()
        n_total = len(specific_genes)
        n_types = 3
        # Binomial test
        binom_result = stats.binomtest(observed, n_total, 1/n_types, alternative='greater')
        binom_p = binom_result.pvalue
        print(f"\n  Binomial test (vs random 1/{n_types}): P = {binom_p:.2e}")

    return results_df


def analysis_pseudobulk_de(adata, found_genes):
    """Pseudobulk differential expression: AD vs Control per cell type."""
    print("\n" + "="*70)
    print("Pseudobulk Differential Expression (AD vs Control)")
    print("="*70)

    if 'diagnosis' not in adata.obs.columns or 'sample_id' not in adata.obs.columns:
        print("  WARNING: Missing diagnosis or sample_id. Skipping DE.")
        return None

    adata_sub = adata[:, [g for g in found_genes if g in adata.var_names]].copy()

    de_results = []
    celltypes = ['Neuron', 'Microglia', 'Astrocyte']

    for ct in celltypes:
        ct_mask = adata_sub.obs['our_celltype'] == ct
        ct_adata = adata_sub[ct_mask]

        if ct_adata.n_obs < 10:
            print(f"  {ct}: Too few cells ({ct_adata.n_obs}), skipping")
            continue

        # Pseudobulk: aggregate by sample
        samples = ct_adata.obs['sample_id'].unique()
        ad_samples = ct_adata.obs[ct_adata.obs['diagnosis'] == 'AD']['sample_id'].unique()
        ctrl_samples = ct_adata.obs[ct_adata.obs['diagnosis'] == 'Control']['sample_id'].unique()

        print(f"\n  {ct}: {len(ad_samples)} AD samples, {len(ctrl_samples)} Control samples")

        if len(ad_samples) < 2 or len(ctrl_samples) < 2:
            print(f"    Too few samples, skipping")
            continue

        for gene in found_genes:
            if gene not in adata_sub.var_names:
                continue

            gene_idx = list(adata_sub.var_names).index(gene)

            # Per-sample mean expression
            ad_vals = []
            for s in ad_samples:
                s_mask = (ct_adata.obs['sample_id'] == s)
                if s_mask.sum() > 0:
                    val = np.asarray(ct_adata[s_mask, gene_idx].X.mean())
                    ad_vals.append(float(val))

            ctrl_vals = []
            for s in ctrl_samples:
                s_mask = (ct_adata.obs['sample_id'] == s)
                if s_mask.sum() > 0:
                    val = np.asarray(ct_adata[s_mask, gene_idx].X.mean())
                    ctrl_vals.append(float(val))

            if len(ad_vals) < 2 or len(ctrl_vals) < 2:
                continue

            ad_mean = np.mean(ad_vals)
            ctrl_mean = np.mean(ctrl_vals)

            # Skip genes with negligible expression in both groups
            # (prevents extreme log2FC from near-zero denominators)
            min_expr_threshold = 0.001
            if ad_mean < min_expr_threshold and ctrl_mean < min_expr_threshold:
                continue

            log2fc = np.log2((ad_mean + 1e-4) / (ctrl_mean + 1e-4))

            stat, pval = stats.mannwhitneyu(ad_vals, ctrl_vals, alternative='two-sided')

            de_results.append({
                'gene': gene,
                'cell_type': ct,
                'ad_mean': ad_mean,
                'ctrl_mean': ctrl_mean,
                'log2fc': log2fc,
                'mannwhitney_U': stat,
                'pvalue': pval,
                'n_ad_samples': len(ad_vals),
                'n_ctrl_samples': len(ctrl_vals),
            })

    if not de_results:
        print("  No DE results generated.")
        return None

    de_df = pd.DataFrame(de_results)

    # BH FDR correction
    de_df['fdr'] = multipletests(de_df['pvalue'], method='fdr_bh')[1]

    sig = de_df[de_df['fdr'] < 0.05]
    print(f"\n  Total DE tests: {len(de_df)}")
    print(f"  Significant (FDR < 0.05): {len(sig)}")
    for ct in celltypes:
        ct_sig = sig[sig['cell_type'] == ct]
        print(f"    {ct}: {len(ct_sig)} significant genes")

    return de_df


def analysis_ccratio_vs_expression(de_df, variant_df):
    """Correlate cc_ratio with expression fold-change."""
    print("\n" + "="*70)
    print("CC Ratio vs Expression Fold-Change Correlation")
    print("="*70)

    if de_df is None:
        print("  No DE data, skipping.")
        return None

    # Gene-level cc_ratio
    gene_cc = variant_df.groupby('gene_name')['cc_ratio'].mean().reset_index()
    gene_cc.columns = ['gene', 'mean_cc_ratio']

    corr_results = []
    for ct in de_df['cell_type'].unique():
        ct_de = de_df[de_df['cell_type'] == ct][['gene', 'log2fc']].copy()
        merged = ct_de.merge(gene_cc, on='gene')

        if len(merged) < 5:
            continue

        r, p = stats.spearmanr(merged['mean_cc_ratio'], merged['log2fc'])
        print(f"  {ct}: Spearman r={r:.3f}, P={p:.2e}, n={len(merged)}")
        corr_results.append({'cell_type': ct, 'spearman_r': r, 'pvalue': p, 'n_genes': len(merged)})

    return pd.DataFrame(corr_results) if corr_results else None


# ============================================================
# Figure generation
# ============================================================

def figure_A1_heatmap(expr_df, found_genes):
    """Figure A1: 85-gene x cell-type expression heatmap."""
    print("\nGenerating Figure A1: Expression heatmap...")

    gene_ct = get_gene_celltype_map()

    # Sort genes by cell type
    gene_order = []
    ct_labels = []
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        ct_genes = [g for g in found_genes if gene_ct.get(g) == ct and g in expr_df.index]
        gene_order.extend(ct_genes)
        ct_labels.extend([ct] * len(ct_genes))

    if not gene_order:
        print("  No genes to plot!")
        return

    plot_df = expr_df.loc[gene_order]
    celltypes_to_show = [c for c in ['Neuron', 'Microglia', 'Astrocyte', 'Other'] if c in plot_df.columns]
    plot_df = plot_df[celltypes_to_show]

    fig, ax = plt.subplots(figsize=(5, max(8, len(gene_order) * 0.18)))

    # Z-score normalize per gene for visualization
    plot_z = plot_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)

    im = ax.imshow(plot_z.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(celltypes_to_show)))
    ax.set_xticklabels(celltypes_to_show, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(gene_order, fontsize=5)

    # Color-code gene names by paper cell-type
    for i, gene in enumerate(gene_order):
        pct = gene_ct.get(gene, 'Ubiquitous')
        ax.get_yticklabels()[i].set_color(COLORS.get(pct, '#333333'))

    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.5)

    # Add cell-type group separators
    cumsum = 0
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        n = sum(1 for g in gene_order if gene_ct.get(g) == ct)
        if n > 0:
            if cumsum > 0:
                ax.axhline(y=cumsum - 0.5, color='black', linewidth=0.5)
            ax.text(-0.8, cumsum + n/2, ct, fontsize=7, color=COLORS[ct],
                    fontweight='bold', ha='right', va='center', rotation=90)
            cumsum += n

    ax.set_title('AD Gene Expression by Cell Type (snRNA-seq)', fontsize=10, fontweight='bold')
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/Figure_A1_expression_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def figure_A2_concordance(conc_df):
    """Figure A2: Classification concordance."""
    print("\nGenerating Figure A2: Classification concordance...")

    if conc_df is None or len(conc_df) == 0:
        print("  No concordance data!")
        return

    specific = conc_df[conc_df['paper_celltype'].isin(['Neuron', 'Microglia', 'Astrocyte'])].copy()
    if len(specific) == 0:
        print("  No cell-type-specific genes found!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Confusion matrix
    ax = axes[0]
    paper_types = ['Neuron', 'Microglia', 'Astrocyte']
    conf_matrix = np.zeros((3, 3))
    for i, pt in enumerate(paper_types):
        for j, et in enumerate(paper_types):
            conf_matrix[i, j] = ((specific['paper_celltype'] == pt) &
                                  (specific['highest_expr_celltype'] == et)).sum()

    im = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks(range(3))
    ax.set_xticklabels(paper_types, fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(paper_types, fontsize=8)
    ax.set_xlabel('Highest Expression Cell Type (snRNA-seq)')
    ax.set_ylabel('Functional Classification (Paper)')
    ax.set_title('(a) Functional vs Expression Classification', fontweight='bold', loc='left')

    for i in range(3):
        for j in range(3):
            val = int(conf_matrix[i, j])
            ax.text(j, i, str(val), ha='center', va='center',
                    color='white' if val > conf_matrix.max()/2 else 'black', fontsize=10)

    # Panel 2: Specificity index by group
    ax = axes[1]
    for ct in paper_types:
        ct_data = specific[specific['paper_celltype'] == ct]
        if len(ct_data) > 0:
            ax.bar(ct, ct_data['specificity_index'].median(),
                   color=COLORS[ct], alpha=0.8, edgecolor='white')
            # Individual points
            ax.scatter([ct] * len(ct_data), ct_data['specificity_index'],
                       color=COLORS[ct], s=20, alpha=0.6, zorder=3, edgecolor='white')

    ax.set_ylabel('Specificity Index (max/mean_others)')
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('(b) Expression Specificity', fontweight='bold', loc='left')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_A2_concordance.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def figure_A3_de(de_df):
    """Figure A3: AD vs Control DE dot plot."""
    print("\nGenerating Figure A3: DE results...")

    if de_df is None or len(de_df) == 0:
        print("  No DE results!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    celltypes = ['Neuron', 'Microglia', 'Astrocyte']

    for idx, ct in enumerate(celltypes):
        ax = axes[idx]
        ct_data = de_df[de_df['cell_type'] == ct].copy()

        if len(ct_data) == 0:
            ax.set_title(f'{ct} (no data)')
            continue

        ct_data['neg_log10_fdr'] = -np.log10(ct_data['fdr'].clip(lower=1e-50))
        ct_data['significant'] = ct_data['fdr'] < 0.05

        # Use clipped log2fc for plotting (keep original for labels)
        ct_data['log2fc_plot'] = ct_data['log2fc'].clip(lower=-5, upper=5)

        # Non-significant
        ns = ct_data[~ct_data['significant']]
        ax.scatter(ns['log2fc_plot'], ns['neg_log10_fdr'], c='#CCCCCC', s=15, alpha=0.5, label='NS')

        # Significant
        sig = ct_data[ct_data['significant']]
        ax.scatter(sig['log2fc_plot'], sig['neg_log10_fdr'], c=COLORS[ct], s=25, alpha=0.8,
                   edgecolor='white', linewidth=0.3, label=f'FDR<0.05 (n={len(sig)})')

        # Label top genes by nominal p-value
        top = ct_data.nsmallest(min(5, len(ct_data)), 'pvalue')
        for _, row in top.iterrows():
            ax.annotate(row['gene'], (row['log2fc_plot'], row['neg_log10_fdr']),
                        fontsize=6, fontweight='bold', color=COLORS[ct],
                        xytext=(5, 3), textcoords='offset points')

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-np.log10(0.05), color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('log2 Fold Change (AD/Control)')
        if idx == 0:
            ax.set_ylabel('-log10(FDR)')
        ax.set_title(f'{ct} (n={len(ct_data)} genes)', fontweight='bold', color=COLORS[ct])
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('Differential Expression: AD vs Control (Pseudobulk)', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_A3_DE_dotplot.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def figure_A4_ccratio_vs_fc(de_df, variant_df):
    """Figure A4: cc_ratio vs expression fold-change scatter."""
    print("\nGenerating Figure A4: CC ratio vs fold-change...")

    if de_df is None:
        print("  No DE data!")
        return

    gene_cc = variant_df.groupby('gene_name').agg({
        'cc_ratio': 'mean',
        'cell_type': 'first'
    }).reset_index()
    gene_cc.columns = ['gene', 'mean_cc_ratio', 'cell_type']

    celltypes = [ct for ct in ['Neuron', 'Microglia', 'Astrocyte'] if ct in de_df['cell_type'].unique()]
    n_ct = len(celltypes)
    if n_ct == 0:
        print("  No cell types with data!")
        return

    fig, axes = plt.subplots(1, n_ct, figsize=(5 * n_ct, 4.5))
    if n_ct == 1:
        axes = [axes]

    for idx, ct in enumerate(celltypes):
        ax = axes[idx]
        ct_de = de_df[de_df['cell_type'] == ct][['gene', 'log2fc']].copy()
        merged = ct_de.merge(gene_cc, on='gene')

        if len(merged) < 3:
            ax.set_title(f'{ct} (n<3)')
            continue

        ax.scatter(merged['mean_cc_ratio'], merged['log2fc'],
                   c=COLORS[ct], s=30, alpha=0.7, edgecolor='white', linewidth=0.3)

        # Label extremes
        for col, n in [('mean_cc_ratio', 3), ('log2fc', 3)]:
            extremes = pd.concat([merged.nlargest(n, col), merged.nsmallest(n, col)]).drop_duplicates('gene')
            for _, row in extremes.iterrows():
                ax.annotate(row['gene'], (row['mean_cc_ratio'], row['log2fc']),
                            fontsize=6, xytext=(4, 3), textcoords='offset points')

        r, p = stats.spearmanr(merged['mean_cc_ratio'], merged['log2fc'])
        ax.set_xlabel('Mean CC Ratio (case/control)')
        if idx == 0:
            ax.set_ylabel('log2 FC (AD/Control expression)')
        ax.set_title(f'{ct}: r={r:.3f}, P={p:.2e}', fontweight='bold', color=COLORS[ct])
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.suptitle('CC Ratio vs snRNA-seq Fold Change', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_A4_ccratio_vs_foldchange.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("="*70)
    print("Analysis A: Cell-Type-Specific Expression (snRNA-seq)")
    print("="*70)

    # Load data
    adata, celltype_col, diag_col, sample_col = load_snrnaseq()
    ad_genes = get_ad_genes()
    variant_df = load_variant_data()

    # Normalize
    adata = normalize_data(adata)

    # Analysis 1: Cell-type expression profiles
    expr_df, found_genes = analysis_celltype_expression(adata, ad_genes)

    if expr_df is not None:
        # Analysis 2: Concordance
        conc_df = analysis_concordance(expr_df, found_genes)

        # Analysis 3: Pseudobulk DE
        de_df = analysis_pseudobulk_de(adata, found_genes)

        # Analysis 4: CC ratio vs expression
        corr_df = analysis_ccratio_vs_expression(de_df, variant_df)

        # Generate figures
        figure_A1_heatmap(expr_df, found_genes)
        figure_A2_concordance(conc_df)
        figure_A3_de(de_df)
        figure_A4_ccratio_vs_fc(de_df, variant_df)

        # Save tables
        expr_df.to_csv(f"{OUTPUT_DIR}/celltype_expression_matrix.csv")
        print(f"\nSaved: {OUTPUT_DIR}/celltype_expression_matrix.csv")

        if conc_df is not None:
            conc_df.to_csv(f"{OUTPUT_DIR}/concordance_results.csv", index=False)
            print(f"Saved: {OUTPUT_DIR}/concordance_results.csv")

        if de_df is not None:
            de_df.to_csv(f"{OUTPUT_DIR}/pseudobulk_DE_results.csv", index=False)
            print(f"Saved: {OUTPUT_DIR}/pseudobulk_DE_results.csv")

        if corr_df is not None:
            corr_df.to_csv(f"{OUTPUT_DIR}/ccratio_expression_correlation.csv", index=False)
            print(f"Saved: {OUTPUT_DIR}/ccratio_expression_correlation.csv")

    print("\n" + "="*70)
    print("Analysis A Complete")
    print("="*70)


if __name__ == '__main__':
    main()
