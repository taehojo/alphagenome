#!/bin/bash
# Download GSE174367 supplementary files from GEO FTP
# Morabito et al. 2021, Nature Genetics - snRNA-seq + snATAC-seq, AD vs Control, prefrontal cortex

set -e

OUTDIR="/N/project/AiLab/alphagenome/data/geo/GSE174367"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174367/suppl"

FILES=(
    "GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"
    "GSE174367_snRNA-seq_cell_meta.csv.gz"
    "GSE174367_snATAC-seq_filtered_peak_bc_matrix.h5"
    "GSE174367_snATAC-seq_cell_meta.csv.gz"
    "GSE174367_bulkRNA_processed.rda.gz"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "Already exists: $f"
    else
        echo "Downloading: $f"
        wget -q --show-progress "${BASE_URL}/${f}"
    fi
done

echo ""
echo "Download complete. File sizes:"
ls -lh "$OUTDIR"
