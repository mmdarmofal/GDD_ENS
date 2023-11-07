# GDD_ENS
Tumor type classifier using cancer genomic panel sequencing data

## Requirements:
* [hg19/gr37 Reference File](https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/)
* [bwa](https://bio-bwa.sourceforge.net/)
* Conda
* python3
* numpy
* pandas
* [PyTorch](https://pytorch.org/)
* [Imbalanced-Learn Library](https://imbalanced-learn.org/stable/index.html)
* [Scopt Library](https://scikit-optimize.github.io/stable/index.html)

## Data Used:
* raw inputs (msk_solid_heme)
   * data_CNA.txt
   * data_mutations_extended.txt
   * data_clinical_patient.txt
   * msk_solid_heme_data_mutations_unfiltered.sigs.tab.txt
   * data_clinical_sample.txt
   * mskimpact_data_cna_hg19.seg
   * data_fusions.txt
* feature table generation 
   * IMPACT505_Gene_list_detailed.xlsx
   * cytoband_table.txt
   * final_hotspot_list.csv
   * feature_annotations.csv
   * ft_colnames.csv
   * fusions.txt
   * tumor_type_final.txt, tumor_type_ordered.csv
* models
   * ensemble.pt
   * ensemble_models.zip - individual MLPs used to generate final model
* GDD-ENS single runs
   * gddnn_kmeans_output.bz2 - shapley values from original model
   * single_ft.csv - template input file
   * single_ft_res.csv - expected output.file    
* adaptive prior
   * prior_table_single.csv, prior_table_multi.csv - template adaptable prior files
   * single_ft_post_prior.csv - template adaptable prior output file


## Workflow:
### GDD-ENS Model Training and Testing
1. Generate feature table
2. Train and Test GDD-ENS Model
   1. Split data into training and testing
   2. Train 10 individual models for classification
   3. Combine 10 models into single ensemble model (GDD-ENS)

### Single GDD-ENS runs
1. Generate feature table and prediction for a single sample
2. Re-scale predictions using prediction-specific adaptable prior (if applicable)

## Quick Links and Tips:
* [Example Workflow](doc/workflow.md)
* If re-training GDD-ENS or training a modified GDD-ENS we recommend using HPC if available and submitting jobs through the bash scripts provided in the [scripts folder](./scripts)
* For any file-based data unloading errors we recommend re-downloading the models/files directly from the GDD-ENS github repo

## Credits:
GDD_ENS was originally written by Madison Darmofal <darmofam@mskcc.org>.

We thank the following people for their extensive assistance in the development of this pipeline:

- Quaid Morris <morrisq@mskcc.org>
- Michael Berger <bergerm1@mskcc.org>
- Shalabh Suman <sumans@mskcc.org>
- Gurnit Atwal <agurnit@gmail.com>

