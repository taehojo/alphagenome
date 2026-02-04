import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('<WORK_DIR>')
OUTPUT_DIR = BASE_DIR / 'analysis/comprehensive_analysis_FULL_1.8M/case_control_86gene'
WORKER_DIR = BASE_DIR / 'worker_results'

MODALITIES = [
    'rna_seq_effect',
    'splice_junctions_effect',
    'splice_sites_effect',
    'splice_site_usage_effect',
    'cage_effect',
    'dnase_effect',
    'chip_histone_effect',
    'chip_tf_effect'
]

def load_alphgenome_results():
    print("=" * 60)
    print("Loading AlphaGenome results...")
    print("=" * 60)

    all_results = []
    pkl_files = sorted(WORKER_DIR.glob('results_*.pkl'))

    for pkl_file in pkl_files:
        print(f"Loading {pkl_file.name}...")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            all_results.extend(data)
            print(f"  - Loaded {len(data):,} variants")

    print(f"\nTotal AlphaGenome variants: {len(all_results):,}")

    ag_df = pd.DataFrame(all_results)
    print(f"Columns: {list(ag_df.columns)}")

    return ag_df

def create_variant_key(row):
    return f"chr{row['chr_num']}:{row['pos']}:{row['REF']}:{row['ALT']}"

def create_variant_key_flipped(row):
    return f"chr{row['chr_num']}:{row['pos']}:{row['ALT']}:{row['REF']}"

def create_pos_key(row):
    return f"chr{row['chr_num']}:{row['pos']}"

def main():
    print("\n" + "=" * 60)
    print("Loading Case-Control Ratio Data...")
    print("=" * 60)

    cc_df = pd.read_csv(OUTPUT_DIR / 'variant_cc_ratios.csv')
    print(f"Total CC ratio variants: {len(cc_df):,}")
    print(f"Unique variants: {cc_df['variant_id'].nunique():,}")

    print("\nCreating variant keys for matching...")
    cc_df['variant_key'] = cc_df.apply(create_variant_key, axis=1)
    cc_df['variant_key_flipped'] = cc_df.apply(create_variant_key_flipped, axis=1)
    cc_df['pos_key'] = cc_df.apply(create_pos_key, axis=1)

    print(f"Sample variant keys from CC data:")
    for i, key in enumerate(cc_df['variant_key'].head(5)):
        flipped = cc_df['variant_key_flipped'].iloc[i]
        print(f"  {key} / flipped: {flipped}")

    ag_df = load_alphgenome_results()

    print(f"\nSample variant_id from AlphaGenome:")
    for vid in ag_df['variant_id'].head(5):
        print(f"  {vid}")

    print("\n" + "=" * 60)
    print("Matching variants (with REF/ALT flip support)...")
    print("=" * 60)

    ag_lookup = {}
    ag_pos_lookup = {}

    for _, row in ag_df.iterrows():
        vid = row['variant_id']
        mods = {mod: row[mod] for mod in MODALITIES if mod in row}
        ag_lookup[vid] = mods

        parts = vid.split(':')
        if len(parts) >= 2:
            pos_key = f"{parts[0]}:{parts[1]}"
            if pos_key not in ag_pos_lookup:
                ag_pos_lookup[pos_key] = []
            ag_pos_lookup[pos_key].append({'vid': vid, 'mods': mods, 'ref': parts[2] if len(parts) > 2 else '', 'alt': parts[3] if len(parts) > 3 else ''})

    print(f"AlphaGenome lookup size: {len(ag_lookup):,}")
    print(f"AlphaGenome position lookup size: {len(ag_pos_lookup):,}")

    unique_cc_keys = cc_df['variant_key'].unique()
    unique_cc_keys_flipped = cc_df['variant_key_flipped'].unique()
    print(f"Unique CC variant keys: {len(unique_cc_keys):,}")

    direct_matched = 0
    flipped_matched = 0
    pos_matched = 0

    for key in unique_cc_keys:
        if key in ag_lookup:
            direct_matched += 1

    for key in unique_cc_keys_flipped:
        if key in ag_lookup:
            flipped_matched += 1

    print(f"\nDirect matches: {direct_matched:,} ({100*direct_matched/len(unique_cc_keys):.2f}%)")
    print(f"Flipped matches: {flipped_matched:,} ({100*flipped_matched/len(unique_cc_keys):.2f}%)")

    print("\n" + "=" * 60)
    print("Mapping AlphaGenome modalities to CC data...")
    print("=" * 60)

    for mod in MODALITIES:
        cc_df[mod] = np.nan
    cc_df['ag_match_type'] = ''

    matched_count = 0
    for idx, row in cc_df.iterrows():
        variant_key = row['variant_key']
        variant_key_flipped = row['variant_key_flipped']
        pos_key = row['pos_key']

        matched = False

        if variant_key in ag_lookup:
            for mod in MODALITIES:
                if mod in ag_lookup[variant_key]:
                    cc_df.at[idx, mod] = ag_lookup[variant_key][mod]
            cc_df.at[idx, 'ag_match_type'] = 'direct'
            matched = True

        elif variant_key_flipped in ag_lookup:
            for mod in MODALITIES:
                if mod in ag_lookup[variant_key_flipped]:
                    cc_df.at[idx, mod] = ag_lookup[variant_key_flipped][mod]
            cc_df.at[idx, 'ag_match_type'] = 'flipped'
            matched = True

        elif pos_key in ag_pos_lookup:
            cc_ref = row['REF']
            cc_alt = row['ALT']
            for ag_var in ag_pos_lookup[pos_key]:
                if (ag_var['ref'] == cc_ref and ag_var['alt'] == cc_alt) or \
                   (ag_var['ref'] == cc_alt and ag_var['alt'] == cc_ref):
                    for mod in MODALITIES:
                        if mod in ag_var['mods']:
                            cc_df.at[idx, mod] = ag_var['mods'][mod]
                    cc_df.at[idx, 'ag_match_type'] = 'position'
                    matched = True
                    break

        if matched:
            matched_count += 1

    print(f"Rows with AlphaGenome data: {matched_count:,} ({100*matched_count/len(cc_df):.2f}%)")
    print(f"\nMatch type breakdown:")
    print(cc_df['ag_match_type'].value_counts())

    print("\nCoverage per modality:")
    for mod in MODALITIES:
        non_null = cc_df[mod].notna().sum()
        print(f"  {mod}: {non_null:,} ({100*non_null/len(cc_df):.2f}%)")

    output_file = OUTPUT_DIR / 'variant_cc_with_alphgenome.csv'
    cc_df.to_csv(output_file, index=False)
    print(f"\nSaved merged data to: {output_file}")

    print("\n" + "=" * 60)
    print("Summary Statistics by Enrichment Category")
    print("=" * 60)

    matched_df = cc_df[cc_df[MODALITIES[0]].notna()].copy()
    print(f"\nAnalyzing {len(matched_df):,} matched variants")

    if len(matched_df) > 0:
        summary_stats = []
        for enrichment in ['case_only', 'case_enriched', 'ctrl_enriched', 'ctrl_only']:
            subset = matched_df[matched_df['enrichment'] == enrichment]
            if len(subset) > 0:
                stats = {'enrichment': enrichment, 'n_variants': len(subset)}
                for mod in MODALITIES:
                    stats[f'{mod}_mean'] = subset[mod].mean()
                    stats[f'{mod}_median'] = subset[mod].median()
                summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)
        print("\nMean AlphaGenome effects by enrichment category:")
        print(summary_df.to_string(index=False))

        summary_df.to_csv(OUTPUT_DIR / 'alphgenome_by_enrichment.csv', index=False)

        print("\n" + "=" * 60)
        print("Gene-level AlphaGenome Summary")
        print("=" * 60)

        gene_summary = []
        for gene in matched_df['gene_name'].unique():
            gene_df = matched_df[matched_df['gene_name'] == gene]
            stats = {
                'gene': gene,
                'n_variants': len(gene_df),
                'n_case_enriched': (gene_df['enrichment'].isin(['case_only', 'case_enriched'])).sum(),
                'n_ctrl_enriched': (gene_df['enrichment'].isin(['ctrl_only', 'ctrl_enriched'])).sum(),
                'mean_cc_ratio': gene_df['cc_ratio'].replace([np.inf, -np.inf], np.nan).mean()
            }
            for mod in MODALITIES:
                stats[f'{mod}_mean'] = gene_df[mod].mean()
            gene_summary.append(stats)

        gene_summary_df = pd.DataFrame(gene_summary)
        gene_summary_df = gene_summary_df.sort_values('mean_cc_ratio', ascending=False)

        print("\nTop 20 genes by mean CC ratio:")
        print(gene_summary_df[['gene', 'n_variants', 'mean_cc_ratio', 'rna_seq_effect_mean', 'splice_junctions_effect_mean']].head(20).to_string(index=False))

        gene_summary_df.to_csv(OUTPUT_DIR / 'alphgenome_by_gene.csv', index=False)

    print("\n" + "=" * 60)
    print("Creating Visualization...")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    if len(matched_df) > 0:
        ax1 = axes[0, 0]
        mod_means = [matched_df[mod].mean() for mod in MODALITIES]
        mod_names = [m.replace('_effect', '').replace('_', ' ') for m in MODALITIES]
        colors = plt.cm.tab10(np.linspace(0, 1, len(MODALITIES)))
        bars = ax1.barh(mod_names, mod_means, color=colors)
        ax1.set_xlabel('Mean Effect Size')
        ax1.set_title('A. Mean AlphaGenome Effects Across All Variants')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax2 = axes[0, 1]
        enrichment_order = ['case_only', 'case_enriched', 'ctrl_enriched', 'ctrl_only']
        x = np.arange(len(enrichment_order))
        width = 0.1

        for i, mod in enumerate(MODALITIES[:4]):
            means = []
            for enr in enrichment_order:
                subset = matched_df[matched_df['enrichment'] == enr]
                means.append(subset[mod].mean() if len(subset) > 0 else 0)
            ax2.bar(x + i*width, means, width, label=mod.replace('_effect', ''))

        ax2.set_xticks(x + width*1.5)
        ax2.set_xticklabels(['Case\nOnly', 'Case\nEnriched', 'Ctrl\nEnriched', 'Ctrl\nOnly'])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylabel('Mean Effect')
        ax2.set_title('B. Key Modalities by Enrichment Category')

        ax3 = axes[1, 0]
        plot_df = matched_df[np.isfinite(matched_df['cc_ratio'])].copy()
        scatter = ax3.scatter(plot_df['cc_ratio'], plot_df['rna_seq_effect'],
                             alpha=0.3, s=10, c='steelblue')
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='CC=1')
        ax3.set_xlabel('Case/Control Ratio')
        ax3.set_ylabel('RNA-seq Effect')
        ax3.set_title('C. CC Ratio vs RNA-seq Effect')
        ax3.set_xlim(0, 5)

        ax4 = axes[1, 1]
        if len(gene_summary_df) > 0:
            top_genes = gene_summary_df.head(20)
            heatmap_data = top_genes[[f'{m}_mean' for m in MODALITIES]].values

            im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdBu_r')
            ax4.set_yticks(range(len(top_genes)))
            ax4.set_yticklabels(top_genes['gene'].values, fontsize=8)
            ax4.set_xticks(range(len(MODALITIES)))
            ax4.set_xticklabels([m.replace('_effect', '').replace('_', '\n') for m in MODALITIES],
                               fontsize=7, rotation=45, ha='right')
            ax4.set_title('D. Top 20 Genes: AlphaGenome Effects')
            plt.colorbar(im, ax=ax4, label='Mean Effect')
    else:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No matched data', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'alphgenome_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {OUTPUT_DIR / 'alphgenome_analysis.png'}")

    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE: AlphaGenome Mapping Summary")
    print("=" * 60)
    print(f"Total CC ratio variants: {len(cc_df):,}")
    print(f"Matched with AlphaGenome: {matched_count:,} ({100*matched_count/len(cc_df):.2f}%)")
    print(f"\nOutput files:")
    print(f"  - variant_cc_with_alphgenome.csv")
    print(f"  - alphgenome_by_enrichment.csv")
    print(f"  - alphgenome_by_gene.csv")
    print(f"  - alphgenome_analysis.png")

if __name__ == '__main__':
    main()
