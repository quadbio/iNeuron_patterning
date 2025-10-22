# iNeuron Morphogens

Code and analysis for Human neuron subtype programming through combinatorial patterning with scRNA-seq readouts.

## Contents
1. Main screen analysis: NGN2 + ASCL1-DLX2, pre and post patterning
    a) Analysis of original morphogen screen (QC+clustering).
    b) Annotation of dataset by mapping with Seurat + marker genes
    c) Analysis of morphogen effects using transcriptional distance and Random Forest models.
    d) Morphogen-GRN networks with pySCENIC.
    e) Analysis of neurotransmitter (NT) and neuropeptide (NP) expression.
    f) Final metadata: Combined cell and cluster annotations for all samples (main_screen_analysis/final_metadata/)
2. Primary atlas
    a) Collection of datasets for primary neuron atlas.
    b) Annotation of primary neuron atlas (NT).
    c) Comparison with iNeurons.
3. Patterned progenitors (single-cell multiome + bulk)
    a) Single-cell multiome analysis of NGN2 patterned progenitors.
    b) Bulk RNA-seq comparison between NGN2 and ASCL1-DLX2 patterned progenitors.
4. Reproducibility + stability
    a) Notebooks for analysis of repdrocibility across cell lines using bulk RNA-seq.
    b) Notebooks for analysis of stability using time-series bulk RNA-seq (d0 to d49).
5. Perturbations of TFs linked to morphogens using OE and KD
    a) Notebooks for analysis of overexpression (OE) experiments
    b) Notebooks for analysis of knock-down (KD) experiments.

## Final Processed Data

### Main Screen Analysis Metadata
Location: `main_screen_analysis/final_metadata/`

Contains the final processed metadata files with combined cell and cluster annotations:
- `GSM7950265_NGN2_post_dr_clustered_raw_merged_meta.tsv.gz` - NGN2 post-patterning metadata
- `GSM7950267_NGN2_pre_dr_clustered_raw_merged_meta.tsv.gz` - NGN2 pre-patterning metadata  
- `GSM7950268_ASCL1_post_dr_clustered_raw_merged_meta.tsv.gz` - ASCL1-DLX2 post-patterning metadata
- `GSM7950269_ASCL1_pre_dr_clustered_raw_merged_meta.tsv.gz` - ASCL1-DLX2 pre-patterning metadata
- `00_Combine_Cell_Cluster_annotation.ipynb` - Notebook for combining annotations

These files contain the complete cell-level metadata including cluster assignments, cell type annotations, morphogen treatments, and quality control metrics for all samples used in the main screen analysis.
