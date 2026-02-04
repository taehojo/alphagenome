#!/bin/bash

BASE_DIR="<WORK_DIR>"
PLINK_DIR="${BASE_DIR}/analysis/plink"
OUTPUT_DIR="${BASE_DIR}/analysis/comprehensive_analysis_FULL_1.8M"
PHENO_FILE="${BASE_DIR}/analysis/comprehensive_analysis_20251202_full/phenotype_full_ids.txt"

echo "========================================"
echo "PLINK Association Analysis - Full 1.8M Variants"
echo "REF/ALT Order Corrected"
echo "========================================"
echo "Start time: $(date)"
echo ""

mkdir -p ${OUTPUT_DIR}/assoc_results

TOTAL_MATCHED=0
TOTAL_RESULTS=0

for CHR in $(seq 1 22); do
    echo "=== Processing chromosome ${CHR} ==="

    EXTRACT_FILE="${OUTPUT_DIR}/variants_chr${CHR}_flipped.extract"
    BED_FILE="${PLINK_DIR}/adsp_chr${CHR}"
    OUTPUT_FILE="${OUTPUT_DIR}/assoc_results/chr${CHR}_assoc"

    if [ ! -f "${EXTRACT_FILE}" ]; then
        echo "  ERROR: Extract file not found: ${EXTRACT_FILE}"
        continue
    fi

    if [ ! -f "${BED_FILE}.bed" ]; then
        echo "  ERROR: BED file not found: ${BED_FILE}.bed"
        continue
    fi

    N_VARIANTS=$(wc -l < ${EXTRACT_FILE})
    echo "  Variants to extract: ${N_VARIANTS}"

    plink --bfile ${BED_FILE} \
          --extract ${EXTRACT_FILE} \
          --pheno ${PHENO_FILE} \
          --fisher \
          --allow-no-sex \
          --out ${OUTPUT_FILE} 2>&1 | grep -E "(variants remaining|pass filters|cases and)"

    if [ -f "${OUTPUT_FILE}.assoc.fisher" ]; then
        N_RESULTS=$(($(wc -l < ${OUTPUT_FILE}.assoc.fisher) - 1))
        echo "  Results: ${N_RESULTS} variants"
        TOTAL_RESULTS=$((TOTAL_RESULTS + N_RESULTS))
    else
        echo "  WARNING: No output file created"
    fi
    echo ""
done

echo "========================================"
echo "Merging all chromosome results..."
echo "========================================"

MERGED_FILE="${OUTPUT_DIR}/all_1.8M_association.assoc.fisher"

head -1 ${OUTPUT_DIR}/assoc_results/chr1_assoc.assoc.fisher > ${MERGED_FILE}

for CHR in $(seq 1 22); do
    ASSOC_FILE="${OUTPUT_DIR}/assoc_results/chr${CHR}_assoc.assoc.fisher"
    if [ -f "${ASSOC_FILE}" ]; then
        tail -n +2 ${ASSOC_FILE} >> ${MERGED_FILE}
    fi
done

N_TOTAL=$(($(wc -l < ${MERGED_FILE}) - 1))
echo ""
echo "========================================"
echo "COMPLETE!"
echo "Total variants analyzed: ${N_TOTAL}"
echo "Output: ${MERGED_FILE}"
echo "End time: $(date)"
echo "========================================"
