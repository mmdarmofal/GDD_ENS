## Initial Set-up:

### Create a Conda Environment using the [requirements](../env/requirements.txt) file:
```
conda env create gdd_ens -f env/requirements.txt
```

### Unzip the Repository Folder:
```
unzip data/msk_solid_heme.zip
```

### Download & Index the Reference Fasta
```
# Download from the UCSC Database:
wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz

# Uncompress the Reference Fasta File
gzip -d hg19.fa.gz

# Create an index of Reference Fasta File
bwa index hg19.fa

``````

## Generate feature table:

   > Inputs: 
   > * path_to_fasta: path to fasta, ex: hg19.fa
   > * repository_folder: folder where data is stored, i.e. data/data_training/msk_solid_heme
   > * label: title of the final output table (if not specified, msk_solid_heme_ft.csv)
   >

   ```
   $ python generate_feature_table.py <path/to/fasta> <path/to/repository> <label>
  ```

   > Outputs:
   > * msk_solid_heme_ft.csv: final data table
   >

## Train and Test GDD-ENS Model:

### 1. Split data into training and testing

   > Inputs: 
   > * label: string specifying if you want to change any of the default file names for the final tables
   >

   ```
   $ python split_data.py <label>
   ```

   > Outputs: 
   > * dropped_cols.csv: any columns that are dropped during training (i.e. all zero values)
   > * ft_train_labelled.csv, ft_test_labelled.csv: feature tables for training/testing with cancer type, sample_id etc
   > * ft_train.csv, ft_test.csv: same feature tables without cancer type, sample_id etc
   > * labels_train.csv, labels_test.csv: cancer types for training/testing
   > * duplicated_patients.csv: any patients with multiple samples that had samples removed
   >

### 2. Train 10 individual models for classification

   > Inputs: 
   > * label: string specifying if you want to change any of the default file names for the final tables/models
   > * fold: which fold of the ensemble model is training / which splits of training and testing to grab (an integer 1-10, this is usually submitted as a bash job array so all 10 submit to HPC and are ran at the same time)
   >

   ```
   $ python train_gdd_nn.py <fold> <label>
   ```

   > Outputs: 
   > * fold_x.pt: final model object, where x is the specified fold of the model
   > * fold_x_params.csv: final model parameters
   >

### 3. Combine 10 models into single ensemble model (GDD-ENS)
   > Inputs: 
   > * label: string specifying if you want to change any of the default file names for the final tables/models
   >

   ```
   $ python gdd_ensemble.py <label>
   ```
   > Outputs: 
   > * ensemble.pt: final ensemble model
   > * ensemble_allprobs.csv: output probabilities for the test set across all cancer types (final outputted layer of the model)
   > * ensemble_results.csv: all output results for the test set (top prediction and probs only)
   >

## Single GDD-ENS runs:
   > Inputs: 
   > * predict_single_fn: filepath to datatable for sample which is being run through GDD
   > * final_res_fn: filepath to save result after running sample through GDD
   >

   ```
   $ python run_gdd_single.py <path/to/predict_single_fn> <path/to/final_res_fn>
   ```

   > Outputs: 
   > * final_res_fn

## Adaptable prior:

> *Requires output of a single run. Compatable with one or more priors, specified in the adaptable prior table with one column (template: prior_table_single)  ore multiple (prior_table_multi)*

   > Inputs: 
   > * prior_file: specified prior, ex: single_rescale_array.csv, multi_rescale_array.csv 
   > * original_output_file: outputted results by GDD-ENS for single sample, ex: template_output_liver.csv
   >

   ```
   $ python adaptable_prior.py <path/to/adaptable_prior_file> <path/to/original_output> <label>
   ```

   > Outputs: 
   > * output_post_prior.csv: new results after adjusting for specified prior(s)
