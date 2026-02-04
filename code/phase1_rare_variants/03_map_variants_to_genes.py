import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from intervaltree import IntervalTree
import pickle

WORK_DIR = "<WORK_DIR>"

PROMOTER_REGION = 5000
UTR3_REGION = 1000

def build_gene_interval_trees():
    print(f"[{datetime.now()}] Building gene interval trees...")

    gene_file = f"{WORK_DIR}/protein_coding_genes.tsv"

    if not os.path.exists(gene_file):
        print(f"ERROR: Gene file not found: {gene_file}")
        print("Please run 00_extract_genes.py first")
        sys.exit(1)

    gene_df = pd.read_csv(gene_file, sep='\t')
    print(f"[{datetime.now()}] Loaded {len(gene_df):,} protein-coding genes")

    chr_trees = {}

    for chr_name in gene_df['chr'].unique():
        chr_trees[chr_name] = IntervalTree()

        chr_genes = gene_df[gene_df['chr'] == chr_name]

        for _, gene in chr_genes.iterrows():
            start = max(1, gene['start'] - PROMOTER_REGION)
            end = gene['end'] + UTR3_REGION

            gene_info = {
                'gene_name': gene['gene_name'],
                'gene_id': gene['gene_id'],
                'strand': gene['strand'],
                'gene_start': gene['start'],
                'gene_end': gene['end'],
                'chr': chr_name
            }

            chr_trees[chr_name][start:end] = gene_info

        print(f"[{datetime.now()}] Chromosome {chr_name}: {len(chr_genes):,} genes indexed")

    tree_file = f"{WORK_DIR}/gene_interval_trees.pkl"
    with open(tree_file, 'wb') as f:
        pickle.dump(chr_trees, f)

    print(f"[{datetime.now()}] Gene interval trees saved to: {tree_file}")

    return chr_trees

def map_variants_to_genes(chr_trees=None):
    print(f"\n[{datetime.now()}] Starting variant-to-gene mapping...")

    if chr_trees is None:
        tree_file = f"{WORK_DIR}/gene_interval_trees.pkl"
        if os.path.exists(tree_file):
            print(f"[{datetime.now()}] Loading pre-built interval trees...")
            with open(tree_file, 'rb') as f:
                chr_trees = pickle.load(f)
        else:
            chr_trees = build_gene_interval_trees()

    variant_file = f"{WORK_DIR}/all_populations_rare_variants.tsv"

    if not os.path.exists(variant_file):
        print(f"[{datetime.now()}] Combined variant file not found, loading individual populations...")

        all_variants = []
        for population in ["AA", "NHW", "Asian", "Hispanic"]:
            pop_file = f"{WORK_DIR}/results/{population}/{population}_rare_variant_list.tsv"
            if os.path.exists(pop_file):
                var_df = pd.read_csv(pop_file, sep='\t')
                var_df['Population'] = population
                all_variants.append(var_df)

        if all_variants:
            variant_df = pd.concat(all_variants, ignore_index=True)
        else:
            print("ERROR: No variant files found")
            sys.exit(1)
    else:
        variant_df = pd.read_csv(variant_file, sep='\t')

    print(f"[{datetime.now()}] Total variants to map: {len(variant_df):,}")

    mapped_variants = []
    unmapped_count = 0

    for idx, variant in variant_df.iterrows():
        if idx % 10000 == 0:
            print(f"[{datetime.now()}] Processed {idx:,} / {len(variant_df):,} variants")

        chr_name = f"chr{variant['CHR']}"
        position = variant['POS']

        if chr_name not in chr_trees:
            unmapped_count += 1
            continue

        overlapping = chr_trees[chr_name][position]

        if overlapping:
            for interval in overlapping:
                gene_info = interval.data

                if position < gene_info['gene_start']:
                    location = 'promoter'
                    distance = gene_info['gene_start'] - position
                elif position > gene_info['gene_end']:
                    location = '3UTR'
                    distance = position - gene_info['gene_end']
                else:
                    location = 'gene_body'
                    distance = 0

                mapped_variant = {
                    'chr': variant['CHR'],
                    'pos': position,
                    'snp_id': variant.get('SNP', f"{chr_name}:{position}"),
                    'ref': variant.get('A1', 'N'),
                    'alt': variant.get('A2', 'N'),
                    'population': variant.get('Population', 'Unknown'),
                    'gene_name': gene_info['gene_name'],
                    'gene_id': gene_info['gene_id'],
                    'strand': gene_info['strand'],
                    'location': location,
                    'distance_to_gene': distance
                }

                mapped_variants.append(mapped_variant)
        else:
            unmapped_count += 1

    mapped_df = pd.DataFrame(mapped_variants)

    print(f"[{datetime.now()}] Mapping complete:")
    print(f"  - Variants mapped to genes: {len(mapped_df):,}")
    print(f"  - Unique variants: {mapped_df[['chr', 'pos']].drop_duplicates().shape[0]:,}")
    print(f"  - Unmapped variants: {unmapped_count:,}")
    print(f"  - Unique genes with variants: {mapped_df['gene_name'].nunique():,}")

    output_file = f"{WORK_DIR}/variants_mapped_to_genes.tsv"
    mapped_df.to_csv(output_file, sep='\t', index=False)
    print(f"[{datetime.now()}] Mapped variants saved to: {output_file}")

    return mapped_df

def analyze_gene_variant_burden():
    print(f"\n[{datetime.now()}] Analyzing gene variant burden...")

    mapped_file = f"{WORK_DIR}/variants_mapped_to_genes.tsv"

    if not os.path.exists(mapped_file):
        print("ERROR: Mapped variants file not found")
        sys.exit(1)

    mapped_df = pd.read_csv(mapped_file, sep='\t')

    gene_burden = mapped_df.groupby('gene_name').agg({
        'pos': 'count',
        'population': lambda x: len(set(x)),
        'location': lambda x: x.value_counts().to_dict()
    }).rename(columns={
        'pos': 'n_variants',
        'population': 'n_populations'
    })

    gene_burden = gene_burden.sort_values('n_variants', ascending=False)

    assoc_file = f"{WORK_DIR}/significant_associations.tsv"
    if os.path.exists(assoc_file):
        assoc_df = pd.read_csv(assoc_file, sep='\t')

        print(f"[{datetime.now()}] Integrating association results...")

        sig_counts = {}
        for gene_name in gene_burden.index:
            gene_variants = mapped_df[mapped_df['gene_name'] == gene_name]

            gene_variant_ids = set()
            for _, var in gene_variants.iterrows():
                var_id = f"{var['chr']}:{var['pos']}"
                gene_variant_ids.add(var_id)

            sig_count = 0
            if not assoc_df.empty and '#CHROM' in assoc_df.columns and 'POS' in assoc_df.columns:
                for _, assoc in assoc_df.iterrows():
                    assoc_id = f"{assoc['#CHROM']}:{assoc['POS']}"
                    if assoc_id in gene_variant_ids:
                        sig_count += 1

            sig_counts[gene_name] = sig_count

        gene_burden['significant_associations'] = pd.Series(sig_counts)

    burden_file = f"{WORK_DIR}/gene_variant_burden.tsv"
    gene_burden.to_csv(burden_file, sep='\t')
    print(f"[{datetime.now()}] Gene burden analysis saved to: {burden_file}")

    print("\n=== Top 20 Genes by Variant Burden ===")
    print(gene_burden.head(20)[['n_variants', 'n_populations']])

    return gene_burden

def prepare_for_alphgenome():
    print(f"\n[{datetime.now()}] Preparing variants for AlphaGenome analysis...")

    mapped_df = pd.read_csv(f"{WORK_DIR}/variants_mapped_to_genes.tsv", sep='\t')

    unique_variants = mapped_df[['chr', 'pos', 'ref', 'alt', 'gene_name', 'gene_id']].drop_duplicates()

    unique_variants['variant_id'] = unique_variants.apply(
        lambda x: f"chr{x['chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1
    )

    ad_genes = ['APP', 'PSEN1', 'PSEN2', 'APOE', 'TREM2', 'SORL1', 'ABCA7', 'CLU', 'CR1',
                'CD33', 'MS4A6A', 'BIN1', 'CD2AP', 'EPHA1', 'PICALM']

    unique_variants['is_ad_gene'] = unique_variants['gene_name'].isin(ad_genes)

    unique_variants = unique_variants.sort_values(
        ['is_ad_gene', 'chr', 'pos'],
        ascending=[False, True, True]
    )

    output_file = f"{WORK_DIR}/variants_for_alphgenome.tsv"
    unique_variants.to_csv(output_file, sep='\t', index=False)

    print(f"[{datetime.now()}] Variants prepared for AlphaGenome:")
    print(f"  - Total unique variants: {len(unique_variants):,}")
    print(f"  - Variants in AD genes: {unique_variants['is_ad_gene'].sum():,}")
    print(f"  - Output file: {output_file}")

    batch_size = 1000
    n_batches = (len(unique_variants) + batch_size - 1) // batch_size

    print(f"\n[{datetime.now()}] Created {n_batches} batches of {batch_size} variants each")
    print(f"Estimated processing time at 1 sec/variant: {len(unique_variants)/3600:.1f} hours")

    return unique_variants

if __name__ == "__main__":
    print("=" * 80)
    print("ADSP Genome-Wide Rare Variant Analysis")
    print("Phase 2: Map Variants to 20,000 Genes")
    print("=" * 80)

    if not os.path.exists(f"{WORK_DIR}/protein_coding_genes.tsv"):
        print("ERROR: Please run 00_extract_genes.py first")
        sys.exit(1)

    chr_trees = build_gene_interval_trees()

    mapped_variants = map_variants_to_genes(chr_trees)

    gene_burden = analyze_gene_variant_burden()

    alphgenome_variants = prepare_for_alphgenome()

    print(f"\n[{datetime.now()}] Phase 2 completed successfully")
    print("Next step: Run 04_alphgenome_analysis.py")