# GDD_ENS
Tumor type classifier using cancer genomic panel sequencing data
## Requirements
* Conda
* python3
* numpy
* pandas
* [PyTorch](https://pytorch.org/)
* [Imbalanced-Learn Library](https://imbalanced-learn.org/stable/index.html)
* [Scopt Library](https://scikit-optimize.github.io/stable/index.html)

## Data
### [data](./data/)
* IMPACT505_Gene_list_detailed.xlsx
* cytoband_table.txt
* ensemble.pt
* feature_annotations.csv
* final_hotspot_list.csv
* ft_colnames.csv
* fusions.txt
* gddnn_kmeans_output.bz2
* msk_solid_heme.zip
* tumor_type_final.txt
* tumor_type_ordered.csv
* msk_solid_heme.zip
  - data_CNA.txt
  - data_clinical_sample.txt
  - data_clinical_patient.txt
  - data_fusions.txt
  - data_mutations_extended.txt
  - msk_solid_heme_data_mutations_unfiltered.sigs.tab.txt
  - mskimpact_data_cna_hg19.seg


## Workflow:
### Generate feature table: 
   ```
   $ python generate_feature_table.py <path/to/fasta> <path/to/repository> <label>
  ```

### Train and Test GDD-ENS Model:
1. Split data into training and testing
   ```
   $ python split_data.py <label>
   ```
2. Train 10 individual models for classification
   ```
   $ python train_gdd_nn.py <fold> <label>
   ```
3. Combine 10 models into single ensemble model (GDD-ENS)
   ```
   $ python gdd_ensemble.py <label>
   ```

### Single GDD-ENS runs:
   ```
   $ python run_gdd_single.py <path/to/single_ft> <path/to/single_output>
   ```
### Adaptable prior:
* Requires original full probability array and outputted formatted as per template. Compatable with one or more priors, specify in adaptable prior table (prior_table_single vs prior_table_multi)
   ```
   python adaptable_prior.py <path/to/adaptable_prior> <path/to/original_output> <path/to/original_allprobs> <label>
   ```
