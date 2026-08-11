"""
Analysis B: Chromatin Accessibility at Variant Sites (snATAC-seq)
================================================================
Validates AlphaGenome regulatory predictions against experimental
chromatin accessibility from GSE174367 snATAC-seq data.

Input: GSE174367 snATAC-seq (130,418 nuclei) + variant data with AlphaGenome scores
Output: Figures B1-B4 + statistical tables

Author: Taeho Jo (tjo@iu.edu)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import h5py
from scipy import stats, sparse
from intervaltree import IntervalTree
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from utils_geo import (PROJECT_DIR, GEO_DIR, RESULTS_BASE, CELL_TYPE_GENES,
                        COLORS, GSE174367_CELLTYPE_MAP, get_gene_celltype_map,
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

OUTPUT_DIR = ensure_dir(f"{RESULTS_BASE}/B")

def load_snATACseq():
    """Load snATAC-seq peak matrix and metadata."""
    print("Loading snATAC-seq data...")

    h5_path = f"{GEO_DIR}/GSE174367_snATAC-seq_filtered_peak_bc_matrix.h5"
    meta_path = f"{GEO_DIR}/GSE174367_snATAC-seq_cell_meta.csv.gz"

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"snATAC h5 file not found: {h5_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cell metadata not found: {meta_path}")

    
    meta = pd.read_csv(meta_path)
    print(f"  Metadata: {len(meta)} rows, columns: {list(meta.columns)}")

    
    print("  Loading peak matrix from h5...")
    with h5py.File(h5_path, 'r') as f:
        
        print(f"  H5 keys: {list(f.keys())}")

        
        if 'matrix' in f:
            grp = f['matrix']
            print(f"  Matrix group keys: {list(grp.keys())}")

            
            data = grp['data'][:]
            indices = grp['indices'][:]
            indptr = grp['indptr'][:]
            shape = grp['shape'][:]

            matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
            print(f"  Matrix shape: {matrix.shape}")

            
            if 'features' in grp:
                feat_grp = grp['features']
                if 'id' in feat_grp:
                    peak_ids = feat_grp['id'][:].astype(str)
                elif 'name' in feat_grp:
                    peak_ids = feat_grp['name'][:].astype(str)
                else:
                    peak_ids = np.array([f"peak_{i}" for i in range(shape[0])])
            elif 'feature_names' in grp:
                peak_ids = grp['feature_names'][:].astype(str)
            else:
                peak_ids = np.array([f"peak_{i}" for i in range(shape[0])])

            
            if 'barcodes' in grp:
                barcodes = grp['barcodes'][:].astype(str)
            else:
                barcodes = np.array([f"cell_{i}" for i in range(shape[1])])

        else:
            
            available = list(f.keys())
            print(f"  Available: {available}")
            raise ValueError(f"Unexpected h5 structure. Keys: {available}")

    print(f"  Peaks: {len(peak_ids)}, Cells: {len(barcodes)}")
    print(f"  Peak examples: {peak_ids[:5]}")
    print(f"  Barcode examples: {barcodes[:3]}")

    return matrix, peak_ids, barcodes, meta

def parse_peak_coordinates(peak_ids):
    """Parse peak IDs into genomic coordinates."""
    print("\nParsing peak coordinates...")

    peaks = []
    for pid in peak_ids:
        pid_str = str(pid)
        try:
            
            if ':' in pid_str:
                chrom, rest = pid_str.split(':')
                start, end = rest.split('-')
            elif pid_str.count('-') == 2:
                parts = pid_str.split('-')
                chrom, start, end = parts[0], parts[1], parts[2]
            elif '_' in pid_str:
                parts = pid_str.split('_')
                if len(parts) >= 3 and parts[0].startswith('chr'):
                    chrom, start, end = parts[0], parts[1], parts[2]
                else:
                    peaks.append(None)
                    continue
            else:
                peaks.append(None)
                continue

            chrom = chrom.replace('chr', '')
            peaks.append({'chr': chrom, 'start': int(start), 'end': int(end), 'peak_id': pid_str})
        except (ValueError, IndexError):
            peaks.append(None)

    valid = sum(1 for p in peaks if p is not None)
    print(f"  Parsed: {valid}/{len(peak_ids)} peaks")
    return peaks

def build_peak_trees(peaks):
    """Build interval trees for peak lookup by chromosome."""
    print("Building interval trees...")
    trees = {}
    for i, p in enumerate(peaks):
        if p is None:
            continue
        chrom = p['chr']
        if chrom not in trees:
            trees[chrom] = IntervalTree()
        trees[chrom][p['start']:p['end']] = i  

    print(f"  Chromosomes: {sorted(trees.keys(), key=lambda x: int(x) if x.isdigit() else 100)}")
    total_peaks = sum(len(t) for t in trees.values())
    print(f"  Total indexed peaks: {total_peaks}")
    return trees

def map_variants_to_peaks(variant_df, peak_trees):
    """Map variants to overlapping ATAC peaks."""
    print("\nMapping variants to ATAC peaks...")

    variant_peak_map = []
    n_mapped = 0

    for _, row in variant_df.iterrows():
        chrom = str(row['chr_num'])
        pos = int(row['pos'])

        if chrom in peak_trees:
            overlaps = peak_trees[chrom][pos]
            if overlaps:
                for interval in overlaps:
                    variant_peak_map.append({
                        'variant_id': row['variant_id'],
                        'peak_idx': interval.data,
                        'gene_name': row['gene_name'],
                        'enrichment': row['enrichment'],
                        'cc_ratio': row['cc_ratio'],
                        'cell_type': row.get('cell_type', 'Unknown'),
                        'dnase_effect': row.get('dnase_effect', np.nan),
                        'chip_histone_effect': row.get('chip_histone_effect', np.nan),
                    })
                n_mapped += 1

    map_df = pd.DataFrame(variant_peak_map) if variant_peak_map else pd.DataFrame()
    unique_mapped = map_df['variant_id'].nunique() if len(map_df) > 0 else 0
    print(f"  Variants overlapping peaks: {unique_mapped}/{len(variant_df)} ({unique_mapped/len(variant_df)*100:.1f}%)")
    print(f"  Total variant-peak pairs: {len(map_df)}")

    return map_df

def assign_cell_metadata(barcodes, meta):
    """Assign cell type and diagnosis to barcodes."""
    print("\nAssigning cell metadata...")

    
    barcode_col = None
    for col in meta.columns:
        if col.lower() == 'barcode':
            barcode_col = col
            break
    if barcode_col is None:
        for col in meta.columns:
            if 'barcode' in col.lower():
                barcode_col = col
                break
    if barcode_col is None:
        barcode_col = meta.columns[0]

    
    celltype_col = None
    for col in meta.columns:
        col_lower = col.lower().replace('.', '_')
        if col_lower == 'cell_type' or col_lower == 'celltype':
            celltype_col = col
            break
    if celltype_col is None:
        for col in meta.columns:
            if col.lower() in ['cluster', 'annotation']:
                celltype_col = col
                break
    if celltype_col is None:
        for col in meta.columns:
            vals = set(meta[col].dropna().astype(str).unique())
            if vals.intersection({'EX', 'INH', 'MG', 'ASC', 'ODC', 'OPC'}):
                celltype_col = col
                break

    
    diag_col = None
    for col in meta.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ['diagnosis', 'disease', 'condition', 'group', 'status']):
            diag_col = col
            break
        vals = set(meta[col].dropna().astype(str).unique())
        if vals.intersection({'AD', 'Control', 'control', 'CTRL'}):
            diag_col = col
            break

    print(f"  Barcode col: {barcode_col}, CellType col: {celltype_col}, Diag col: {diag_col}")

    
    meta_indexed = meta.set_index(barcode_col)

    celltype_map = {}
    diag_map = {}

    if celltype_col and celltype_col in meta_indexed.columns:
        for bc, val in meta_indexed[celltype_col].items():
            val_str = str(val).strip()
            
            if val_str in GSE174367_CELLTYPE_MAP:
                celltype_map[str(bc)] = GSE174367_CELLTYPE_MAP[val_str]
            else:
                
                major = val_str.split('-')[0].split('_')[0].split('.')[0].strip()
                celltype_map[str(bc)] = GSE174367_CELLTYPE_MAP.get(major, 'Other')

    if diag_col and diag_col in meta_indexed.columns:
        for bc, val in meta_indexed[diag_col].items():
            v_str = str(val).lower()
            if 'ad' in v_str or 'alzheimer' in v_str:
                diag_map[str(bc)] = 'AD'
            elif 'control' in v_str or 'ctrl' in v_str or 'normal' in v_str:
                diag_map[str(bc)] = 'Control'

    
    cell_ct = []
    cell_diag = []
    for bc in barcodes:
        bc_str = str(bc)
        bc_stripped = bc_str.rsplit('-', 1)[0]
        cell_ct.append(celltype_map.get(bc_str, celltype_map.get(bc_stripped, 'Unknown')))
        cell_diag.append(diag_map.get(bc_str, diag_map.get(bc_stripped, 'Unknown')))

    ct_counts = pd.Series(cell_ct).value_counts()
    print(f"\n  Cell type assignments:")
    for ct, n in ct_counts.items():
        print(f"    {ct}: {n:,}")

    return np.array(cell_ct), np.array(cell_diag)

def compute_pseudobulk_accessibility(matrix, cell_types, variant_peak_map):
    """Compute pseudobulk accessibility per cell type at variant sites."""
    print("\nComputing pseudobulk accessibility...")

    if len(variant_peak_map) == 0:
        print("  No variant-peak mappings!")
        return None

    
    matrix_csr = matrix.tocsr() if not sparse.issparse(matrix) else matrix.tocsr()

    peak_indices = variant_peak_map['peak_idx'].unique()

    results = []
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Other']:
        ct_mask = cell_types == ct
        n_cells = ct_mask.sum()
        if n_cells == 0:
            continue

        
        ct_matrix = matrix_csr[:, ct_mask]

        for pidx in peak_indices:
            if pidx >= matrix_csr.shape[0]:
                continue
            
            peak_access = ct_matrix[pidx, :].mean()

            
            peak_variants = variant_peak_map[variant_peak_map['peak_idx'] == pidx]
            for _, vrow in peak_variants.iterrows():
                results.append({
                    'variant_id': vrow['variant_id'],
                    'gene_name': vrow['gene_name'],
                    'enrichment': vrow['enrichment'],
                    'cc_ratio': vrow['cc_ratio'],
                    'variant_celltype': vrow['cell_type'],
                    'dnase_effect': vrow['dnase_effect'],
                    'accessibility_celltype': ct,
                    'mean_accessibility': float(peak_access),
                })

    access_df = pd.DataFrame(results)
    print(f"  Accessibility records: {len(access_df)}")
    return access_df

def analysis_open_chromatin_enrichment(variant_df, variant_peak_map):
    """Compare open chromatin overlap: case-enriched vs control-enriched."""
    print("\n" + "="*70)
    print("Open Chromatin Enrichment (Case vs Control)")
    print("="*70)

    if len(variant_peak_map) == 0:
        print("  No mapping data!")
        return None

    
    mapped_variants = set(variant_peak_map['variant_id'].unique())

    case_variants = variant_df[variant_df['enrichment'] == 'case_enriched']
    ctrl_variants = variant_df[variant_df['enrichment'] == 'ctrl_enriched']

    a = len(set(case_variants['variant_id']) & mapped_variants)
    b = len(case_variants) - a
    c = len(set(ctrl_variants['variant_id']) & mapped_variants)
    d = len(ctrl_variants) - c

    case_rate = a / (a + b) if (a + b) > 0 else 0
    ctrl_rate = c / (c + d) if (c + d) > 0 else 0

    odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])

    print(f"  Case-enriched in open chromatin: {a}/{a+b} ({case_rate:.1%})")
    print(f"  Ctrl-enriched in open chromatin: {c}/{c+d} ({ctrl_rate:.1%})")
    print(f"  Fisher's exact OR = {odds_ratio:.4f}, P = {fisher_p:.4e}")

    return {
        'case_in_peak': a, 'case_not_in_peak': b,
        'ctrl_in_peak': c, 'ctrl_not_in_peak': d,
        'case_rate': case_rate, 'ctrl_rate': ctrl_rate,
        'odds_ratio': odds_ratio, 'fisher_p': fisher_p
    }

def analysis_alphgenome_vs_accessibility(access_df):
    """Correlate AlphaGenome dnase_effect with experimental accessibility."""
    print("\n" + "="*70)
    print("AlphaGenome vs Experimental Accessibility Correlation")
    print("="*70)

    if access_df is None or len(access_df) == 0:
        print("  No accessibility data!")
        return None

    results = []
    for ct in ['Neuron', 'Microglia', 'Astrocyte']:
        ct_data = access_df[access_df['accessibility_celltype'] == ct].dropna(subset=['dnase_effect', 'mean_accessibility'])
        if len(ct_data) < 5:
            continue
        r, p = stats.spearmanr(ct_data['dnase_effect'], ct_data['mean_accessibility'])
        print(f"  {ct}: Spearman r={r:.3f}, P={p:.2e}, n={len(ct_data)}")
        results.append({'cell_type': ct, 'spearman_r': r, 'pvalue': p, 'n': len(ct_data)})

    
    valid = access_df.dropna(subset=['dnase_effect', 'mean_accessibility'])
    if len(valid) >= 5:
        r, p = stats.spearmanr(valid['dnase_effect'], valid['mean_accessibility'])
        print(f"  Overall: Spearman r={r:.3f}, P={p:.2e}, n={len(valid)}")
        results.append({'cell_type': 'Overall', 'spearman_r': r, 'pvalue': p, 'n': len(valid)})

    return pd.DataFrame(results) if results else None

def figure_B1_chromatin_overlap(enrichment_result):
    """Figure B1: Open chromatin overlap rate."""
    print("\nGenerating Figure B1: Chromatin overlap...")

    if enrichment_result is None:
        print("  No data!")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    categories = ['Case-enriched', 'Control-enriched']
    rates = [enrichment_result['case_rate'] * 100, enrichment_result['ctrl_rate'] * 100]
    colors_bar = ['#E64B35', '#4DBBD5']

    bars = ax.bar(categories, rates, color=colors_bar, alpha=0.85, edgecolor='white', width=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

    or_val = enrichment_result['odds_ratio']
    p_val = enrichment_result['fisher_p']
    ax.set_title(f"Open Chromatin Overlap\nFisher's OR={or_val:.3f}, P={p_val:.2e}",
                 fontweight='bold', fontsize=10)
    ax.set_ylabel('% Variants in ATAC Peaks')
    ax.set_ylim(0, max(rates) * 1.2)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_B1_chromatin_overlap.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

def figure_B2_celltype_accessibility(access_df):
    """Figure B2: Cell-type-specific accessibility at variant sites."""
    print("\nGenerating Figure B2: Cell-type accessibility violin...")

    if access_df is None or len(access_df) == 0:
        print("  No data!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    
    ax = axes[0]
    ct_order = [ct for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Other']
                if ct in access_df['accessibility_celltype'].unique()]

    plot_data = access_df[access_df['accessibility_celltype'].isin(ct_order)]
    if len(plot_data) > 0:
        parts = ax.violinplot(
            [plot_data[plot_data['accessibility_celltype'] == ct]['mean_accessibility'].values
             for ct in ct_order],
            positions=range(len(ct_order)),
            showmeans=True, showmedians=True
        )
        ax.set_xticks(range(len(ct_order)))
        ax.set_xticklabels(ct_order, fontsize=8)
        ax.set_ylabel('Mean Accessibility')
        ax.set_title('(a) By Cell Type', fontweight='bold', loc='left')

    
    ax = axes[1]
    for enr, color, label in [('case_enriched', '#E64B35', 'Case'), ('ctrl_enriched', '#4DBBD5', 'Control')]:
        enr_data = access_df[access_df['enrichment'] == enr]
        if len(enr_data) > 0:
            for i, ct in enumerate(ct_order):
                ct_enr = enr_data[enr_data['accessibility_celltype'] == ct]['mean_accessibility']
                offset = -0.15 if enr == 'case_enriched' else 0.15
                bp = ax.boxplot(ct_enr.values, positions=[i + offset], widths=0.25,
                                patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.6)
                bp['medians'][0].set_color('black')

    ax.set_xticks(range(len(ct_order)))
    ax.set_xticklabels(ct_order, fontsize=8)
    ax.set_ylabel('Mean Accessibility')
    ax.set_title('(b) Case vs Control Enriched', fontweight='bold', loc='left')

    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#E64B35', alpha=0.6, label='Case-enriched'),
                       Patch(facecolor='#4DBBD5', alpha=0.6, label='Ctrl-enriched')],
              fontsize=7, loc='upper right')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_B2_celltype_accessibility.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

def figure_B3_alphgenome_vs_atac(access_df):
    """Figure B3: AlphaGenome score vs experimental accessibility."""
    print("\nGenerating Figure B3: AlphaGenome vs ATAC...")

    if access_df is None or len(access_df) == 0:
        print("  No data!")
        return

    if 'dnase_effect' not in access_df.columns or 'mean_accessibility' not in access_df.columns:
        print("  Missing required columns!")
        return

    valid = access_df.dropna(subset=['dnase_effect', 'mean_accessibility'])
    if len(valid) < 5:
        print("  Too few data points!")
        return

    celltypes = [ct for ct in ['Neuron', 'Microglia', 'Astrocyte']
                 if ct in valid['accessibility_celltype'].unique()]
    n_ct = max(len(celltypes), 1)

    fig, axes = plt.subplots(1, n_ct, figsize=(5 * n_ct, 4.5))
    if n_ct == 1:
        axes = [axes]

    for idx, ct in enumerate(celltypes):
        ax = axes[idx]
        ct_data = valid[valid['accessibility_celltype'] == ct]

        if len(ct_data) < 3:
            ax.set_title(f'{ct} (n<3)')
            continue

        ax.scatter(ct_data['dnase_effect'], ct_data['mean_accessibility'],
                   c=COLORS.get(ct, '#666666'), s=15, alpha=0.5, edgecolor='white', linewidth=0.2)

        r, p = stats.spearmanr(ct_data['dnase_effect'], ct_data['mean_accessibility'])
        ax.set_xlabel('AlphaGenome DNase Effect')
        if idx == 0:
            ax.set_ylabel('Experimental ATAC Accessibility')
        ax.set_title(f'{ct}: r={r:.3f}, P={p:.2e}', fontweight='bold', color=COLORS.get(ct, '#333'))

        
        if len(ct_data) >= 10:
            z = np.polyfit(ct_data['dnase_effect'], ct_data['mean_accessibility'], 1)
            x_line = np.linspace(ct_data['dnase_effect'].min(), ct_data['dnase_effect'].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), '--', color=COLORS.get(ct, '#666'), alpha=0.5, linewidth=1)

    plt.suptitle('AlphaGenome Prediction vs Experimental Accessibility', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_B3_alphgenome_vs_atac.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

def figure_B4_gene_accessibility_heatmap(access_df):
    """Figure B4: Gene-level cell-type accessibility heatmap."""
    print("\nGenerating Figure B4: Gene accessibility heatmap...")

    if access_df is None or len(access_df) == 0:
        print("  No data!")
        return

    
    pivot = access_df.pivot_table(
        values='mean_accessibility',
        index='gene_name',
        columns='accessibility_celltype',
        aggfunc='mean'
    )

    celltypes = [ct for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Other'] if ct in pivot.columns]
    pivot = pivot[celltypes]

    
    gene_ct = get_gene_celltype_map()
    pivot['paper_ct'] = pivot.index.map(lambda g: gene_ct.get(g, 'Ubiquitous'))
    ct_order = {'Neuron': 0, 'Microglia': 1, 'Astrocyte': 2, 'Ubiquitous': 3}
    pivot['sort_key'] = pivot['paper_ct'].map(ct_order)
    pivot = pivot.sort_values('sort_key')

    fig, ax = plt.subplots(figsize=(5, max(6, len(pivot) * 0.2)))

    plot_data = pivot[celltypes]
    
    plot_z = plot_data.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)

    im = ax.imshow(plot_z.values, aspect='auto', cmap='YlOrRd', vmin=-2, vmax=2)
    ax.set_xticks(range(len(celltypes)))
    ax.set_xticklabels(celltypes, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=5)

    
    for i, gene in enumerate(pivot.index):
        pct = gene_ct.get(gene, 'Ubiquitous')
        ax.get_yticklabels()[i].set_color(COLORS.get(pct, '#333333'))

    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.5)
    ax.set_title('Chromatin Accessibility by Cell Type', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/Figure_B4_gene_accessibility_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

def main():
    print("="*70)
    print("Analysis B: Chromatin Accessibility (snATAC-seq)")
    print("="*70)

    
    variant_df = load_variant_data()
    print(f"Variant data: {len(variant_df)} unique variants")

    
    matrix, peak_ids, barcodes, meta = load_snATACseq()

    
    peaks = parse_peak_coordinates(peak_ids)

    
    peak_trees = build_peak_trees(peaks)

    
    variant_peak_map = map_variants_to_peaks(variant_df, peak_trees)

    
    cell_types, cell_diag = assign_cell_metadata(barcodes, meta)

    
    enrichment_result = analysis_open_chromatin_enrichment(variant_df, variant_peak_map)

    
    access_df = None
    if len(variant_peak_map) > 0:
        access_df = compute_pseudobulk_accessibility(matrix, cell_types, variant_peak_map)

    
    corr_df = analysis_alphgenome_vs_accessibility(access_df)

    
    figure_B1_chromatin_overlap(enrichment_result)
    figure_B2_celltype_accessibility(access_df)
    figure_B3_alphgenome_vs_atac(access_df)
    figure_B4_gene_accessibility_heatmap(access_df)

    
    if enrichment_result:
        pd.DataFrame([enrichment_result]).to_csv(f"{OUTPUT_DIR}/chromatin_enrichment.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/chromatin_enrichment.csv")

    if variant_peak_map is not None and len(variant_peak_map) > 0:
        variant_peak_map.to_csv(f"{OUTPUT_DIR}/variant_peak_mapping.csv", index=False)
        print(f"Saved: {OUTPUT_DIR}/variant_peak_mapping.csv")

    if access_df is not None:
        access_df.to_csv(f"{OUTPUT_DIR}/celltype_accessibility.csv", index=False)
        print(f"Saved: {OUTPUT_DIR}/celltype_accessibility.csv")

    if corr_df is not None:
        corr_df.to_csv(f"{OUTPUT_DIR}/alphgenome_atac_correlation.csv", index=False)
        print(f"Saved: {OUTPUT_DIR}/alphgenome_atac_correlation.csv")

    print("\n" + "="*70)
    print("Analysis B Complete")
    print("="*70)

if __name__ == '__main__':
    main()
