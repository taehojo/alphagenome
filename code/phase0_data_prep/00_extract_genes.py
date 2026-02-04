import gzip
import pandas as pd
import os
import sys
from datetime import datetime

WORK_DIR = "<WORK_DIR>"
GENE_ANNOTATION = f"{WORK_DIR}/gencode.v38.annotation.gtf.gz"

def extract_protein_coding_genes():
    print(f"[{datetime.now()}] Starting protein-coding gene extraction from GENCODE v38")
    print(f"[{datetime.now()}] Input file: {GENE_ANNOTATION}")

    if not os.path.exists(GENE_ANNOTATION):
        print(f"ERROR: GENCODE annotation file not found: {GENE_ANNOTATION}")
        print("Please ensure gencode.v38.annotation.gtf.gz is in the working directory")
        sys.exit(1)

    genes = []
    processed_lines = 0

    print(f"[{datetime.now()}] Reading compressed GTF file...")

    with gzip.open(GENE_ANNOTATION, 'rt') as f:
        for line in f:
            processed_lines += 1

            if processed_lines % 100000 == 0:
                print(f"[{datetime.now()}] Processed {processed_lines:,} lines, found {len(genes):,} protein-coding genes")

            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')

            if len(fields) < 9:
                continue

            if fields[2] == 'gene':
                info = fields[8]

                if 'gene_type "protein_coding"' in info or 'gene_biotype "protein_coding"' in info:
                    gene_name = None
                    gene_id = None

                    for attr in info.split(';'):
                        attr = attr.strip()
                        if attr.startswith('gene_name'):
                            gene_name = attr.split('"')[1]
                        elif attr.startswith('gene_id'):
                            gene_id = attr.split('"')[1].split('.')[0]

                    if gene_name and gene_id:
                        genes.append({
                            'gene_name': gene_name,
                            'gene_id': gene_id,
                            'chr': fields[0],
                            'start': int(fields[3]),
                            'end': int(fields[4]),
                            'strand': fields[6],
                            'length': int(fields[4]) - int(fields[3]) + 1
                        })

    print(f"[{datetime.now()}] Extraction complete. Total lines processed: {processed_lines:,}")
    print(f"[{datetime.now()}] Total protein-coding genes found: {len(genes):,}")

    gene_df = pd.DataFrame(genes)

    gene_df = gene_df.drop_duplicates(subset=['gene_id'])
    print(f"[{datetime.now()}] Unique protein-coding genes after deduplication: {len(gene_df):,}")

    chr_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    gene_df['chr_num'] = pd.Categorical(gene_df['chr'], categories=chr_order, ordered=True)
    gene_df = gene_df.sort_values(['chr_num', 'start'])
    gene_df = gene_df.drop('chr_num', axis=1)

    output_file = f'{WORK_DIR}/protein_coding_genes.tsv'
    gene_df.to_csv(output_file, sep='\t', index=False)
    print(f"[{datetime.now()}] Results saved to: {output_file}")

    print("\n=== Summary Statistics ===")
    print(f"Total protein-coding genes: {len(gene_df):,}")
    print("\nGenes per chromosome:")
    chr_counts = gene_df['chr'].value_counts().sort_index()
    for chr_name, count in chr_counts.items():
        if chr_name in chr_order:
            print(f"  {chr_name}: {count:,}")

    print(f"\nAverage gene length: {gene_df['length'].mean():,.0f} bp")
    print(f"Median gene length: {gene_df['length'].median():,.0f} bp")
    print(f"Longest gene: {gene_df.loc[gene_df['length'].idxmax(), 'gene_name']} ({gene_df['length'].max():,} bp)")
    print(f"Shortest gene: {gene_df.loc[gene_df['length'].idxmin(), 'gene_name']} ({gene_df['length'].min():,} bp)")

    return gene_df

if __name__ == "__main__":
    print("=" * 80)
    print("ADSP Genome-Wide Rare Variant Analysis")
    print("Phase 0: Extract Protein-Coding Genes from GENCODE v38")
    print("=" * 80)

    gene_df = extract_protein_coding_genes()

    print("\n[{datetime.now()}] Phase 0 completed successfully")
    print("Next step: Run 01_process_phenotypes.py")