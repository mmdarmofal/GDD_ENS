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
* [data_training](./doc/data.md#data_training)
* [msk solid heme](./doc/data.md#msk_solid_heme)
* [data_model](./doc/data.md#data_model)
* [data_adaptivePrior](./doc/data.md#data_adaptiveprior)

## Workflow:
### GDD-ENS Model Training and Testing
1. Generate feature table
2. Train and Test GDD-ENS Model
   1. Split data into training and testing
   2. Train 10 individual models for classification
   3. Combine 10 models into single ensemble model (GDD-ENS)

### Single GDD-ENS runs
3. Generate prediction from a single sample
4. Re-scale predictions using prediction-specific adaptable prior (if applicable)

## Quick Links and Tips:
* [Example Workflow](doc/workflow.md)
* If re-training GDD-ENS or training a modified GDD-ENS we recommend using HPC if available and submitting jobs through the bash scripts provided in the [scripts folder](./doc/scripts.md#scripts)
* For any file-based data unloading errors we recommend re-downloading the models/files directly from this repo and re-running

## Credits:
GDD_ENS was originally written by Madison Darmofal <darmofam@mskcc.org>.

We thank the following people for their extensive assistance in the development of this pipeline:

- Quaid Morris <morrisq@mskcc.org>
- Michael Berger <bergerm1@mskcc.org>
- Shalabh Suman <sumans@mskcc.org>
- Gurnit Atwal <agurnit@gmail.com>

