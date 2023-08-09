# GDD_ENS
Tumor type classifier using cancer genomic panel sequencing data
## Requirements
- requirements.txt
## Data
- data
  - tumor_type_final.txt
  - tumor_type_ordered.csv
  - IMPACT505_Gene_list_detailed.xlsx
  - cytoband_table.txt
  - final_hotspot_list.csv
  - fusions.txt
  - ft_colnames.csv
  - gddnn_kmeans_output.bz2
  - feature_annotations.csv
- msk_solid_heme.zip
  - data_clinical_sample.txt
  - data_clinical_patient.txt
  - data_mutations_extended.txt
  - msk_solid_heme_data_mutations_unfiltered.sigs.tab.txt
  - data_CNA.txt
  - mskimpact_data_cna_hg19.seg
  - data_fusions.txt

## Generate feature table: 
- `$ python generate_feature_table.py <path/to/fasta> <path/to/repository> <label>`

## Train and Test GDD-ENS Model: 
1. split data into training and testing
- `$ python split_data.py <label>`
2. train 10 individual models for classification
- `$ python train_gdd_nn.py <fold> <label>`
3. combine 10 models into single ensemble model (GDD-ENS)
- `$ python gdd_ensemble.py <label>`

## Single GDD-ENS runs: 
`$ python run_gdd_single.py <path/to/single_ft> <path/to/single_output>`

## Adaptable prior
#### *Requires original full probability array and outputted formatted as per template. Compatable with one or more priors, specify in adaptable prior table (prior_table_single vs prior_table_multi)
`$ python adaptable_prior.py <path/to/adaptable_prior> <path/to/original_output> <path/to/original_allprobs> <label>`

