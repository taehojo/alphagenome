#!/usr/bin/env python3
"""
Shared utilities for GEO validation analyses (Phase 6).
"""

import pandas as pd
import numpy as np
import os

PROJECT_DIR = "/N/project/AiLab/alphagenome"
GEO_DIR = f"{PROJECT_DIR}/data/geo/GSE174367"
VARIANT_DATA = f"{PROJECT_DIR}/data/variant_cc_with_alphgenome.csv"
GENE_LIST = f"{PROJECT_DIR}/data/Table_S1_gene_list.csv"
RESULTS_BASE = f"{PROJECT_DIR}/results/geo_validation"

# Cell-type gene mapping (from figure4, used in the manuscript)
CELL_TYPE_GENES = {
    'Neuron': ['APP', 'PSEN1', 'PSEN2', 'MAPT', 'BIN1', 'CLU', 'SORL1', 'ANK3', 'PTK2B', 'ADAM10',
               'APH1B', 'FERMT2', 'SLC24A4', 'CASS4', 'PICALM', 'CD2AP', 'EPHA1'],
    'Microglia': ['TREM2', 'PLCG2', 'ABI3', 'INPP5D', 'SPI1', 'CD33', 'MS4A4A', 'MS4A6A',
                  'CR1', 'PILRA', 'LILRB2', 'TREML2', 'SCIMP', 'CLNK', 'BLNK'],
    'Astrocyte': ['CLU', 'APOE', 'ABCA7', 'ABCA1', 'GRN', 'CTSH', 'CTSB'],
    'Ubiquitous': ['ADAM17', 'JAZF1', 'UMAD1', 'RHOH', 'RASGEF1C', 'HS3ST5', 'SNX1',
                   'PLEKHA1', 'WDR81', 'WDR12', 'MINDY2', 'TSPAN14', 'EPDR1', 'NCK2',
                   'TMEM106B', 'SPPL2A', 'EED', 'ACE', 'TPCN1', 'MME', 'ICA1', 'SORT1',
                   'ANKH', 'FOXF1', 'USP6NL', 'IDUA', 'KLF16', 'COX7C', 'SPDYE3',
                   'RBCK1', 'SHARPIN', 'TNIP1', 'TSPOAP1', 'CASP7', 'PRKD3', 'WNT3',
                   'HLA-DQA1', 'MYO15A', 'PRDM7', 'RIN3', 'ADAMTS1', 'IL34', 'DOC2A',
                   'APBB3', 'SIGLEC11', 'BCKDK', 'UNC5CL', 'SEC61G', 'MAF', 'SLC2A4RG']
}

COLORS = {
    'Microglia': '#E64B35',
    'Neuron': '#00A087',
    'Astrocyte': '#4DBBD5',
    'Ubiquitous': '#8C8C8C',
}

# GSE174367 cell-type mapping to our categories
GSE174367_CELLTYPE_MAP = {
    'EX': 'Neuron',
    'INH': 'Neuron',
    'MG': 'Microglia',
    'ASC': 'Astrocyte',
    'ODC': 'Other',
    'OPC': 'Other',
    'PER': 'Other',
    'END': 'Other',
    'FIB': 'Other',
    'PER.END': 'Other',
    'PER/END': 'Other',
}


def get_gene_celltype_map():
    """Return gene -> cell_type dict (first assignment wins for dual-mapped genes like CLU)."""
    gene_ct = {}
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        for g in CELL_TYPE_GENES[ct]:
            if g not in gene_ct:
                gene_ct[g] = ct
    return gene_ct


def get_ad_genes():
    """Return set of 85 AD gene names from Table S1."""
    gene_df = pd.read_csv(GENE_LIST)
    return set(gene_df['gene_name'].tolist())


def load_variant_data():
    """Load and deduplicate variant data (AC>=3, 9,943 unique variants)."""
    df = pd.read_csv(VARIANT_DATA)
    df['total_AC'] = df['case_AC'] + df['ctrl_AC']
    df = df[df['total_AC'] >= 3].copy()
    df = df.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')

    gene_ct = get_gene_celltype_map()
    df['cell_type'] = df['gene_name'].map(gene_ct)

    return df


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path
