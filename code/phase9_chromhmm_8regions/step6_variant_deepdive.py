#!/usr/bin/env python3
"""
Phase 7 Step 6: Top 20 Variant Deep-Dive

For each of the 20 selected variants, create a mechanistic profile using:
1. Full AlphaGenome track data (HDF5 from Step 3)
2. chromHMM annotations (from Step 5)
3. Cell-type-specific regulatory impact analysis

Outputs per-variant profiles and summary figures.
"""

import pandas as pd
import numpy as np
import h5py
import json
import os
import io
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

BASE = '/N/project/AiLab/alphagenome'
HDF5_DIR = f'{BASE}/results/phase7_rerun/raw'
CHROMHMM_FILE = f'{BASE}/results/phase7_chromhmm/variants_chromhmm_annotated.csv'
TOP20_FILE = f'{BASE}/results/phase7_rerun/top20_variants_for_rerun.csv'
OUT_DIR = f'{BASE}/results/phase7_deepdive'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/variant_profiles', exist_ok=True)

# Modalities of interest
KEY_MODALITIES = ['dnase', 'chip_histone', 'chip_tf', 'cage', 'rna_seq', 'atac']

# Brain cell type labels
CELLTYPE_LABELS = {
    'UBERON:0000955': 'Brain (whole)',
    'CL:0000540': 'Neuron',
    'CL:0000127': 'Astrocyte',
}


def load_hdf5_data(variant_id):
    """Load full track data from HDF5 for a variant."""
    # Sanitize variant_id for filename
    safe_id = variant_id.replace(':', '_').replace('/', '_')
    h5_path = f'{HDF5_DIR}/{safe_id}.h5'

    if not os.path.exists(h5_path):
        print(f"  WARNING: HDF5 not found for {variant_id}")
        return None

    data = {}
    with h5py.File(h5_path, 'r') as f:
        for modality in f.keys():
            mod_data = {}
            # Load ref and alt arrays
            if 'ref' in f[modality] and 'alt' in f[modality]:
                ref_arr = f[modality]['ref']['data'][:]
                alt_arr = f[modality]['alt']['data'][:]
                mod_data['ref'] = ref_arr
                mod_data['alt'] = alt_arr
                mod_data['diff'] = alt_arr - ref_arr

                # Load metadata
                if 'metadata_csv' in f[modality].attrs:
                    meta_csv = f[modality].attrs['metadata_csv']
                    mod_data['metadata'] = pd.read_csv(io.StringIO(meta_csv))

            data[modality] = mod_data

    return data


def analyze_tf_binding(data):
    """Analyze TF binding changes from chip_tf modality."""
    if 'chip_tf' not in data or 'metadata' not in data['chip_tf']:
        return pd.DataFrame()

    meta = data['chip_tf']['metadata']
    diff = data['chip_tf']['diff']

    # Max absolute effect per track (across positions)
    max_effects = np.max(np.abs(diff), axis=0)
    # Also get signed max effect (at position of max abs)
    signed_effects = []
    for i in range(diff.shape[1]):
        idx = np.argmax(np.abs(diff[:, i]))
        signed_effects.append(diff[idx, i])

    results = meta.copy()
    results['max_abs_effect'] = max_effects
    results['signed_max_effect'] = signed_effects

    # Filter to brain-relevant cell types
    brain_ontologies = set(CELLTYPE_LABELS.keys())
    if 'ontology_curie' in results.columns:
        results = results[results['ontology_curie'].isin(brain_ontologies)]

    # Group by TF
    if 'transcription_factor' in results.columns and len(results) > 0:
        tf_summary = results.groupby('transcription_factor').agg(
            max_effect=('max_abs_effect', 'max'),
            mean_effect=('max_abs_effect', 'mean'),
            n_celltypes=('ontology_curie', 'nunique'),
        ).sort_values('max_effect', ascending=False)
        return tf_summary.head(20)

    return pd.DataFrame()


def analyze_histone_marks(data):
    """Analyze histone mark changes from chip_histone modality."""
    if 'chip_histone' not in data or 'metadata' not in data['chip_histone']:
        return pd.DataFrame()

    meta = data['chip_histone']['metadata']
    diff = data['chip_histone']['diff']

    max_effects = np.max(np.abs(diff), axis=0)
    signed_effects = []
    for i in range(diff.shape[1]):
        idx = np.argmax(np.abs(diff[:, i]))
        signed_effects.append(diff[idx, i])

    results = meta.copy()
    results['max_abs_effect'] = max_effects
    results['signed_max_effect'] = signed_effects

    # Filter to brain
    brain_ontologies = set(CELLTYPE_LABELS.keys())
    if 'ontology_curie' in results.columns:
        results = results[results['ontology_curie'].isin(brain_ontologies)]

    if 'histone_mark' in results.columns and len(results) > 0:
        mark_summary = results.groupby(['histone_mark', 'ontology_curie']).agg(
            max_effect=('max_abs_effect', 'max'),
            signed_effect=('signed_max_effect', lambda x: x.iloc[np.argmax(np.abs(x))]),
        ).sort_values('max_effect', ascending=False)
        return mark_summary

    return pd.DataFrame()


def analyze_accessibility(data):
    """Analyze DNase/ATAC accessibility changes."""
    results = {}
    for mod in ['dnase', 'atac']:
        if mod not in data or 'metadata' not in data[mod]:
            continue

        meta = data[mod]['metadata']
        diff = data[mod]['diff']

        max_effects = np.max(np.abs(diff), axis=0)
        signed_effects = []
        for i in range(diff.shape[1]):
            idx = np.argmax(np.abs(diff[:, i]))
            signed_effects.append(diff[idx, i])

        r = meta.copy()
        r['max_abs_effect'] = max_effects
        r['signed_max_effect'] = signed_effects

        brain_ontologies = set(CELLTYPE_LABELS.keys())
        if 'ontology_curie' in r.columns:
            r = r[r['ontology_curie'].isin(brain_ontologies)]

        if len(r) > 0:
            r['celltype'] = r['ontology_curie'].map(CELLTYPE_LABELS)
            results[mod] = r.groupby('celltype').agg(
                max_effect=('max_abs_effect', 'max'),
                mean_effect=('max_abs_effect', 'mean'),
                signed_max=('signed_max_effect', lambda x: x.iloc[np.argmax(np.abs(x))]),
            )

    return results


def analyze_expression(data):
    """Analyze RNA-seq and CAGE expression changes."""
    results = {}
    for mod in ['rna_seq', 'cage']:
        if mod not in data or 'metadata' not in data[mod]:
            continue

        meta = data[mod]['metadata']
        diff = data[mod]['diff']

        max_effects = np.max(np.abs(diff), axis=0)
        signed_effects = []
        for i in range(diff.shape[1]):
            idx = np.argmax(np.abs(diff[:, i]))
            signed_effects.append(diff[idx, i])

        r = meta.copy()
        r['max_abs_effect'] = max_effects
        r['signed_max_effect'] = signed_effects

        brain_ontologies = set(CELLTYPE_LABELS.keys())
        if 'ontology_curie' in r.columns:
            r = r[r['ontology_curie'].isin(brain_ontologies)]

        if len(r) > 0:
            r['celltype'] = r['ontology_curie'].map(CELLTYPE_LABELS)
            results[mod] = r.groupby('celltype').agg(
                max_effect=('max_abs_effect', 'max'),
                mean_effect=('max_abs_effect', 'mean'),
                signed_max=('signed_max_effect', lambda x: x.iloc[np.argmax(np.abs(x))]),
            )

    return results


def create_variant_profile(variant_row, data, chromhmm_row=None):
    """Create a comprehensive profile for one variant."""
    vid = variant_row['variant_id']
    profile = {
        'variant_id': vid,
        'gene': variant_row['gene_name'],
        'cell_type': variant_row['cell_type'],
        'chr': variant_row['chr_num'],
        'pos': variant_row['pos'],
        'ref': variant_row['REF'],
        'alt': variant_row['ALT'],
        'cc_ratio': variant_row['cc_ratio'],
        'case_AC': variant_row['case_AC'],
        'ctrl_AC': variant_row['ctrl_AC'],
    }

    if data is None:
        profile['error'] = 'HDF5 data not found'
        return profile

    # Available modalities
    profile['modalities_available'] = list(data.keys())

    # TF analysis
    tf_results = analyze_tf_binding(data)
    if len(tf_results) > 0:
        profile['top_TFs'] = []
        for tf_name, tf_row in tf_results.head(10).iterrows():
            profile['top_TFs'].append({
                'TF': tf_name,
                'max_effect': float(tf_row['max_effect']),
                'mean_effect': float(tf_row['mean_effect']),
            })

    # Histone marks
    histone_results = analyze_histone_marks(data)
    if len(histone_results) > 0:
        profile['histone_marks'] = []
        for (mark, onto), h_row in histone_results.head(10).iterrows():
            profile['histone_marks'].append({
                'mark': mark,
                'celltype': CELLTYPE_LABELS.get(onto, onto),
                'max_effect': float(h_row['max_effect']),
                'direction': 'gain' if float(h_row['signed_effect']) > 0 else 'loss',
            })

    # Accessibility
    access_results = analyze_accessibility(data)
    if access_results:
        profile['accessibility'] = {}
        for mod, mod_df in access_results.items():
            profile['accessibility'][mod] = {}
            for ct, ct_row in mod_df.iterrows():
                profile['accessibility'][mod][ct] = {
                    'max_effect': float(ct_row['max_effect']),
                    'direction': 'open' if float(ct_row['signed_max']) > 0 else 'close',
                }

    # Expression
    expr_results = analyze_expression(data)
    if expr_results:
        profile['expression'] = {}
        for mod, mod_df in expr_results.items():
            profile['expression'][mod] = {}
            for ct, ct_row in mod_df.iterrows():
                profile['expression'][mod][ct] = {
                    'max_effect': float(ct_row['max_effect']),
                    'direction': 'up' if float(ct_row['signed_max']) > 0 else 'down',
                }

    # chromHMM states
    if chromhmm_row is not None:
        profile['chromhmm_states'] = {}
        epigenome_names = {
            'E067': 'Angular Gyrus', 'E068': 'Anterior Caudate',
            'E069': 'Cingulate Gyrus', 'E071': 'Hippocampus Middle',
            'E072': 'Inferior Temporal Lobe', 'E073': 'DLPFC',
            'E074': 'Substantia Nigra', 'E125': 'NH-A Astrocytes',
        }
        for eid, ename in epigenome_names.items():
            col = f'state_{eid}'
            if col in chromhmm_row.index and pd.notna(chromhmm_row[col]):
                profile['chromhmm_states'][ename] = chromhmm_row[col]
        profile['n_regulatory_epigenomes'] = int(chromhmm_row.get('n_regulatory_epigenomes', 0))
        profile['n_active_regulatory_epigenomes'] = int(chromhmm_row.get('n_active_regulatory_epigenomes', 0))

    return profile


def create_variant_figure(profile, data, variant_idx):
    """Create a multi-panel figure for one variant."""
    vid = profile['variant_id']
    safe_id = vid.replace(':', '_').replace('/', '_')

    if data is None:
        return

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Variant Deep-Dive: {vid}\n"
                 f"Gene: {profile['gene']} ({profile['cell_type']}) | "
                 f"cc_ratio: {profile['cc_ratio']:.2f} | "
                 f"Case AC: {profile['case_AC']}, Ctrl AC: {profile['ctrl_AC']}",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # Panel 1: Top TFs affected
    ax1 = fig.add_subplot(gs[0, 0])
    if 'top_TFs' in profile and len(profile['top_TFs']) > 0:
        tf_df = pd.DataFrame(profile['top_TFs'])
        colors = ['#e74c3c' if e > 0 else '#3498db' for e in tf_df['max_effect']]
        ax1.barh(range(len(tf_df)), tf_df['max_effect'], color=colors)
        ax1.set_yticks(range(len(tf_df)))
        ax1.set_yticklabels(tf_df['TF'], fontsize=8)
        ax1.set_xlabel('Max |Effect|')
        ax1.set_title('Top TF Binding Changes', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No TF data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('TF Binding Changes', fontweight='bold')

    # Panel 2: Histone marks
    ax2 = fig.add_subplot(gs[0, 1])
    if 'histone_marks' in profile and len(profile['histone_marks']) > 0:
        h_df = pd.DataFrame(profile['histone_marks'])
        h_df['label'] = h_df['mark'] + '\n(' + h_df['celltype'] + ')'
        colors = ['#e74c3c' if d == 'gain' else '#3498db' for d in h_df['direction']]
        ax2.barh(range(len(h_df)), h_df['max_effect'], color=colors)
        ax2.set_yticks(range(len(h_df)))
        ax2.set_yticklabels(h_df['label'], fontsize=7)
        ax2.set_xlabel('Max |Effect|')
        ax2.set_title('Histone Mark Changes', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No histone data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Histone Mark Changes', fontweight='bold')

    # Panel 3: chromHMM states
    ax3 = fig.add_subplot(gs[0, 2])
    if 'chromhmm_states' in profile and len(profile['chromhmm_states']) > 0:
        states = profile['chromhmm_states']
        regions = list(states.keys())
        state_vals = list(states.values())

        # Color by state type
        state_colors = {
            '1_TssA': '#FF0000', '2_TssAFlnk': '#FF6969',
            '3_TxFlnk': '#00A600', '4_Tx': '#009600', '5_TxWk': '#88C288',
            '6_EnhG': '#FFCC00', '7_Enh': '#FFFF00',
            '8_ZNF/Rpts': '#66CDAA', '9_Het': '#8A91D0',
            '10_TssBiv': '#CD5C5C', '11_BivFlnk': '#E9967A',
            '12_EnhBiv': '#BDB76B', '13_ReprPC': '#808080',
            '14_ReprPCWk': '#C0C0C0', '15_Quies': '#FFFFFF',
        }
        colors = [state_colors.get(s, '#999999') for s in state_vals]

        ax3.barh(range(len(regions)), [1]*len(regions), color=colors, edgecolor='black', linewidth=0.5)
        ax3.set_yticks(range(len(regions)))
        ax3.set_yticklabels(regions, fontsize=8)
        for i, sv in enumerate(state_vals):
            ax3.text(0.5, i, sv, ha='center', va='center', fontsize=7)
        ax3.set_xlim(0, 1)
        ax3.set_xticks([])
        ax3.set_title(f'chromHMM States\n(Reg: {profile.get("n_regulatory_epigenomes", "?")} / '
                      f'Active: {profile.get("n_active_regulatory_epigenomes", "?")})', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No chromHMM data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('chromHMM States', fontweight='bold')

    # Panel 4-5: DNase effect across positions (ref vs alt)
    ax4 = fig.add_subplot(gs[1, :2])
    if 'dnase' in data and 'metadata' in data['dnase']:
        meta = data['dnase']['metadata']
        brain_mask = meta['ontology_curie'].isin(CELLTYPE_LABELS.keys()) if 'ontology_curie' in meta.columns else pd.Series([True]*len(meta))
        brain_indices = meta[brain_mask].index.tolist()

        if brain_indices:
            # Plot mean ref and alt across positions for brain cell types
            ref_brain = data['dnase']['ref'][:, brain_indices].mean(axis=1)
            alt_brain = data['dnase']['alt'][:, brain_indices].mean(axis=1)
            positions = np.arange(len(ref_brain))

            ax4.plot(positions, ref_brain, color='#3498db', alpha=0.8, label='Reference', linewidth=1)
            ax4.plot(positions, alt_brain, color='#e74c3c', alpha=0.8, label='Alternate', linewidth=1)
            ax4.axvline(x=len(positions)//2, color='black', linestyle='--', alpha=0.3, label='Variant position')
            ax4.fill_between(positions, ref_brain, alt_brain, alpha=0.2, color='grey')
            ax4.set_xlabel('Genomic position (bins)')
            ax4.set_ylabel('DNase signal (mean brain)')
            ax4.set_title('DNase Accessibility: Ref vs Alt (Brain)', fontweight='bold')
            ax4.legend(fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No brain DNase tracks', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No DNase data', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('DNase Accessibility: Ref vs Alt', fontweight='bold')

    # Panel 6: Accessibility by cell type
    ax5 = fig.add_subplot(gs[1, 2])
    if 'accessibility' in profile:
        ct_labels = []
        ct_effects = []
        ct_dirs = []
        for mod in ['dnase', 'atac']:
            if mod in profile['accessibility']:
                for ct, vals in profile['accessibility'][mod].items():
                    ct_labels.append(f"{ct}\n({mod})")
                    ct_effects.append(vals['max_effect'])
                    ct_dirs.append(vals['direction'])
        if ct_labels:
            colors = ['#e74c3c' if d == 'open' else '#3498db' for d in ct_dirs]
            ax5.barh(range(len(ct_labels)), ct_effects, color=colors)
            ax5.set_yticks(range(len(ct_labels)))
            ax5.set_yticklabels(ct_labels, fontsize=8)
            ax5.set_xlabel('Max |Effect|')
    ax5.set_title('Accessibility by Cell Type', fontweight='bold')

    # Panel 7-8: RNA-seq effect across positions
    ax6 = fig.add_subplot(gs[2, :2])
    if 'rna_seq' in data and 'metadata' in data['rna_seq']:
        meta = data['rna_seq']['metadata']
        brain_mask = meta['ontology_curie'].isin(CELLTYPE_LABELS.keys()) if 'ontology_curie' in meta.columns else pd.Series([True]*len(meta))
        brain_indices = meta[brain_mask].index.tolist()

        if brain_indices:
            ref_brain = data['rna_seq']['ref'][:, brain_indices].mean(axis=1)
            alt_brain = data['rna_seq']['alt'][:, brain_indices].mean(axis=1)
            positions = np.arange(len(ref_brain))

            ax6.plot(positions, ref_brain, color='#3498db', alpha=0.8, label='Reference', linewidth=1)
            ax6.plot(positions, alt_brain, color='#e74c3c', alpha=0.8, label='Alternate', linewidth=1)
            ax6.axvline(x=len(positions)//2, color='black', linestyle='--', alpha=0.3)
            ax6.fill_between(positions, ref_brain, alt_brain, alpha=0.2, color='grey')
            ax6.set_xlabel('Genomic position (bins)')
            ax6.set_ylabel('RNA-seq signal (mean brain)')
            ax6.legend(fontsize=8)
    ax6.set_title('RNA-seq Expression: Ref vs Alt (Brain)', fontweight='bold')

    # Panel 9: Expression by cell type
    ax7 = fig.add_subplot(gs[2, 2])
    if 'expression' in profile:
        ct_labels = []
        ct_effects = []
        ct_dirs = []
        for mod in ['rna_seq', 'cage']:
            if mod in profile['expression']:
                for ct, vals in profile['expression'][mod].items():
                    ct_labels.append(f"{ct}\n({mod})")
                    ct_effects.append(vals['max_effect'])
                    ct_dirs.append(vals['direction'])
        if ct_labels:
            colors = ['#e74c3c' if d == 'up' else '#3498db' for d in ct_dirs]
            ax7.barh(range(len(ct_labels)), ct_effects, color=colors)
            ax7.set_yticks(range(len(ct_labels)))
            ax7.set_yticklabels(ct_labels, fontsize=8)
            ax7.set_xlabel('Max |Effect|')
    ax7.set_title('Expression Change by Cell Type', fontweight='bold')

    plt.savefig(f'{OUT_DIR}/variant_profiles/{safe_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_figure(all_profiles):
    """Create a summary comparison across all 20 variants."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Phase 7: Top 20 Variant Deep-Dive Summary', fontsize=16, fontweight='bold')

    # Collect data for summary
    variants = []
    for p in all_profiles:
        if 'error' in p:
            continue
        v = {
            'variant': f"{p['gene']}\n{p['variant_id'][:20]}",
            'gene': p['gene'],
            'cc_ratio': p['cc_ratio'],
            'cell_type': p['cell_type'],
        }

        # Max DNase effect
        if 'accessibility' in p and 'dnase' in p['accessibility']:
            v['dnase_max'] = max(d['max_effect'] for d in p['accessibility']['dnase'].values())
        else:
            v['dnase_max'] = 0

        # Max expression effect
        if 'expression' in p:
            effects = []
            for mod in p['expression']:
                for ct in p['expression'][mod]:
                    effects.append(p['expression'][mod][ct]['max_effect'])
            v['expr_max'] = max(effects) if effects else 0
        else:
            v['expr_max'] = 0

        # Number of TFs affected
        v['n_TFs'] = len(p.get('top_TFs', []))

        # Regulatory epigenomes
        v['n_reg'] = p.get('n_regulatory_epigenomes', 0)

        variants.append(v)

    if not variants:
        print("No variant data available for summary figure")
        return

    vdf = pd.DataFrame(variants)

    # Panel 1: DNase effect by variant
    ax = axes[0, 0]
    ct_colors = {'Neuron': '#2ecc71', 'Microglia': '#e74c3c', 'Astrocyte': '#3498db', 'Ubiquitous': '#95a5a6'}
    colors = [ct_colors.get(ct, '#999999') for ct in vdf['cell_type']]
    vdf_sorted = vdf.sort_values('dnase_max', ascending=True)
    idx_sorted = vdf_sorted.index
    ax.barh(range(len(vdf_sorted)), vdf_sorted['dnase_max'],
            color=[colors[i] for i in idx_sorted])
    ax.set_yticks(range(len(vdf_sorted)))
    ax.set_yticklabels(vdf_sorted['variant'], fontsize=7)
    ax.set_xlabel('Max DNase Effect')
    ax.set_title('DNase Accessibility Impact', fontweight='bold')

    # Panel 2: Expression effect by variant
    ax = axes[0, 1]
    vdf_sorted2 = vdf.sort_values('expr_max', ascending=True)
    idx_sorted2 = vdf_sorted2.index
    ax.barh(range(len(vdf_sorted2)), vdf_sorted2['expr_max'],
            color=[colors[i] for i in idx_sorted2])
    ax.set_yticks(range(len(vdf_sorted2)))
    ax.set_yticklabels(vdf_sorted2['variant'], fontsize=7)
    ax.set_xlabel('Max Expression Effect')
    ax.set_title('Expression Impact', fontweight='bold')

    # Panel 3: cc_ratio vs regulatory impact
    ax = axes[1, 0]
    for ct in ct_colors:
        mask = vdf['cell_type'] == ct
        if mask.any():
            ax.scatter(vdf[mask]['cc_ratio'], vdf[mask]['dnase_max'],
                      color=ct_colors[ct], label=ct, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Case-Control Ratio')
    ax.set_ylabel('Max DNase Effect')
    ax.set_title('Case Enrichment vs Regulatory Impact', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 4: Cell type legend + summary table
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "SUMMARY STATISTICS\n" + "="*40 + "\n\n"
    summary_text += f"Total variants profiled: {len(variants)}\n"
    summary_text += f"Mean cc_ratio: {vdf['cc_ratio'].mean():.1f}\n"
    summary_text += f"Mean DNase effect: {vdf['dnase_max'].mean():.1f}\n"
    summary_text += f"Mean expression effect: {vdf['expr_max'].mean():.1f}\n\n"
    summary_text += "Cell type breakdown:\n"
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        n = (vdf['cell_type'] == ct).sum()
        if n > 0:
            mean_d = vdf[vdf['cell_type'] == ct]['dnase_max'].mean()
            summary_text += f"  {ct}: n={n}, mean DNase={mean_d:.1f}\n"

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/summary_deepdive.png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("STEP 6: Top 20 Variant Deep-Dive")
    print("=" * 60)

    # Load top 20 variants
    top20 = pd.read_csv(TOP20_FILE)
    print(f"Top 20 variants loaded: {len(top20)}")

    # Load chromHMM annotations
    try:
        chromhmm = pd.read_csv(CHROMHMM_FILE)
        chromhmm = chromhmm.set_index('variant_id')
        print(f"chromHMM annotations loaded: {len(chromhmm)} variants")
    except Exception as e:
        print(f"WARNING: Could not load chromHMM: {e}")
        chromhmm = None

    # Process each variant
    all_profiles = []
    for idx, row in top20.iterrows():
        vid = row['variant_id']
        print(f"\n[{idx+1}/20] Processing {vid} ({row['gene_name']})...")

        # Load HDF5 data
        data = load_hdf5_data(vid)

        # Get chromHMM annotation
        chm_row = chromhmm.loc[vid] if chromhmm is not None and vid in chromhmm.index else None

        # Create profile
        profile = create_variant_profile(row, data, chm_row)
        all_profiles.append(profile)

        # Create individual figure
        create_variant_figure(profile, data, idx)

        # Print key findings
        if 'top_TFs' in profile and len(profile['top_TFs']) > 0:
            top_tf = profile['top_TFs'][0]
            print(f"  Top TF: {top_tf['TF']} (effect={top_tf['max_effect']:.2f})")
        if 'accessibility' in profile and 'dnase' in profile['accessibility']:
            for ct, vals in profile['accessibility']['dnase'].items():
                print(f"  DNase {ct}: {vals['direction']} ({vals['max_effect']:.2f})")
        if 'chromhmm_states' in profile:
            print(f"  chromHMM regulatory: {profile.get('n_regulatory_epigenomes', 0)}/8 epigenomes")

    # Save all profiles as JSON
    with open(f'{OUT_DIR}/all_profiles.json', 'w') as f:
        json.dump(all_profiles, f, indent=2, default=str)

    # Create summary figure
    create_summary_figure(all_profiles)

    # Save summary table
    summary_rows = []
    for p in all_profiles:
        row = {
            'variant_id': p['variant_id'],
            'gene': p['gene'],
            'cell_type': p['cell_type'],
            'cc_ratio': p['cc_ratio'],
        }
        if 'top_TFs' in p and len(p['top_TFs']) > 0:
            row['top_TF'] = p['top_TFs'][0]['TF']
            row['top_TF_effect'] = p['top_TFs'][0]['max_effect']
        if 'accessibility' in p and 'dnase' in p['accessibility']:
            for ct, vals in p['accessibility']['dnase'].items():
                row[f'dnase_{ct}'] = vals['max_effect']
                row[f'dnase_{ct}_dir'] = vals['direction']
        if 'expression' in p and 'rna_seq' in p['expression']:
            for ct, vals in p['expression']['rna_seq'].items():
                row[f'rnaseq_{ct}'] = vals['max_effect']
                row[f'rnaseq_{ct}_dir'] = vals['direction']
        row['n_regulatory_epigenomes'] = p.get('n_regulatory_epigenomes', 0)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{OUT_DIR}/summary_table.csv', index=False)

    # Log
    log = {
        'step': 6,
        'status': 'SUCCESS',
        'n_variants_profiled': len(all_profiles),
        'n_with_tf_data': sum(1 for p in all_profiles if 'top_TFs' in p),
        'n_with_chromhmm': sum(1 for p in all_profiles if 'chromhmm_states' in p),
        'output_dir': OUT_DIR,
    }

    log_path = f'{BASE}/results/phase7_logs/step_6.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STEP 6 COMPLETE")
    print(f"{'='*60}")
    print(f"Variants profiled: {log['n_variants_profiled']}")
    print(f"With TF data: {log['n_with_tf_data']}")
    print(f"With chromHMM: {log['n_with_chromhmm']}")
    print(f"Profiles: {OUT_DIR}/all_profiles.json")
    print(f"Summary: {OUT_DIR}/summary_table.csv")
    print(f"Figures: {OUT_DIR}/variant_profiles/")

    return log


if __name__ == '__main__':
    main()
