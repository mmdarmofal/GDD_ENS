## Initial Set-up:

### Download and install [git-lfs](https://git-lfs.com/) for accessing large files:
```
conda install -c conda-forge git-lfs 
git lfs install 
git clone https://github.com/mmdarmofal/GDD_ENS.git
```

### Create a Conda Environment using the [requirements](../env/gdd_ens_env.yml) yml file:
```
conda env create gdd_ens_env -f env/gdd_ens_env.yml
```

### Unzip the Repository Folder and Individual Models:
```
unzip data/msk_solid_heme.zip
unzip data/ensemble_models.zip
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

## Generate full feature table:

   > Inputs: 
   > * path_to_fasta: path to fasta, ex: hg19.fa
   > * path_to_ft: path to output feature table, ex: output/msk_solid_heme_ft.csv

   ```
   $ python scripts/generate_feature_table.py <path/to/fasta> <path/to/ft>
  ```

   > Outputs:
   > * final data table with specified filename as per path_to_ft
   >

## Train and Test GDD-ENS Model:

### 1. Split data into training and testing

   > Inputs: 
   > * path_to_ft: path to output feature table, ex: output/msk_solid_heme_ft.csv


   ```
   $ python scripts/split_data.py <path/to/ft>
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
   > * fold: which fold of the ensemble model is training / which splits of training and testing to grab (an integer 1-10, this is usually submitted as a bash job array so all 10 submit to HPC and are ran at the same time, otherwise needs to be run ten separate times for generation of the ten independent models)
   >

   ```
   $ python scripts/train_gdd_nn.py <fold> 
   ```

   > Outputs: 
   > * fold_x.pt: final model object, where x is the specified fold of the model
   > * fold_x_params.csv: final model parameters
   >

### 3. Combine 10 models into single ensemble model (GDD-ENS)
   > Inputs: 
   > * pre_load: string indicating where to load and save models: "True" if loading directly from the original model (stored in data/ensemble_models), "False" if re-trained or re-ran (stored in output)
   >

   ```
   $ python scripts/gdd_ensemble.py 'True'
   ```
   > Outputs: 
   > * ensemble.pt: final ensemble model
   > * ensemble_allprobs.csv: output probabilities for the test set across all cancer types (final outputted layer of the model)
   > * ensemble_results.csv: all output results for the test set (top prediction and probs only)
   >

## Single GDD-ENS runs:
   > Inputs:
   > * model: filepath to GDD model
   > * input_table_fn: filepath to datatable for sample which is being run through GDD (required column = 'SAMPLE_ID')
   > * final_res_fn: filepath to save result after running sample through GDD
   >

   ```
   $ python scripts/run_gdd_single.py <path/to/model> <path/to/input_table_fn> <path/to/final_res_fn>
   ```

   > Outputs: 
   > * final_res_fn

## Adaptable prior:

> *Requires output of a single run. Compatable with one or more priors, specified in the adaptable prior table with one column (template: prior_table_single)  ore multiple (prior_table_multi)*

   > Inputs: 
   > * prior_file: specified prior, ex: prior_table_single.csv, prior_table_multi.csv
   > * original_output_file: outputted results by GDD-ENS for single sample, ex: single_ft_res.csv
   > * new_output_file: filename for results after adjusting for the prior

   ```
   $ python scripts/adaptable_prior.py <path/to/adaptable_prior_file> <path/to/original_output> <path/to/new_output>
   ```

   > Outputs: 
   > * new_output_file: results after adjusting for one or more priors new_pred and new_prob correspond to the new top prediction and confidence
>    * new_output_rescale_array: full final output including resulting confidence values across all cancer types
