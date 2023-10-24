# GDD_ENS
Tumor type classifier using cancer genomic panel sequencing data
## Requirements:
* [hg19/gr37 Reference File](https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/)
* Conda
* python3
* numpy
* pandas
* [PyTorch](https://pytorch.org/)
* [Imbalanced-Learn Library](https://imbalanced-learn.org/stable/index.html)
* [Scopt Library](https://scikit-optimize.github.io/stable/index.html)

## Data Used:
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
1. Generate feature table
2. Train and Test GDD-ENS Model
   1. Split data into training and testing
   2. Train 10 individual models for classification
   3. Combine 10 models into single ensemble model (GDD-ENS)
3. Single GDD-ENS runs
4. Adaptable prior

## Quick Links:
* [Example Workflow](doc/workflow.md)

## Credits:
GDD_ENS was originally written by Madison Darmofal <darmofam@mskcc.org>.

We thank the following people for their extensive assistance in the development of this pipeline:

- Quaid Morris <morrisq@mskcc.org>
- Michael Berger <bergerm1@mskcc.org>
- Shalabh Suman <sumans@mskcc.org>

