## Instructions for re-training and comparing GDD-RF to GDD-ENS
#### Load data and additional requirements
1.  Clone GDD-RF repo (<https://github.com/bergerm1/GenomeDerivedDiagnosis>)
2.  Install GDD-RF package

    ```{linux GDD_RF_install}
    R CMD INSTALL GenomeDerivedDiagnosis
    ```
    - note: majority of required scripts have been modified from their original versions, all are present in modified_scripts folder, still require internal GenomeDerivedDiagnosis functions
    - instructions for loading required R packages if not already installed are in generate_feature_table_mod.R

3.  Unzip inst/example_data/msk_impact_2017.zip, comparison_data.zip and preloaded_results.zip
#### First comparison: GDD-ENS architecture on GDD-RF train and extended test sets
4.  Generate GDD-RF feature table
    -   **path_to_repo** - path to where input data is stored (ex: "inst/example_data/msk_impact_2017")

    -   **feature_table_filename** - filename for outputted feature table (ex: 'gdd_rf_orig_dt.csv')

    ```{linux generate_ft}
    Rscript modified_scripts/generate_feature_table_mod.R path/to/repo feature_table_filename
    ```

5.  Train ensemble model on GDD-RF feature table

    -   **train_filename** - path to where input training data is stored (i.e. comparison_data/gdd_rf_orig_dt.csv)

    -   **test_filename** - path to where input testing data is stored (i.e. comparison_data/ft_test_labelled_orig_ext.csv)

    -   **fold** - integer 1-10, representing split of data used for testing

        -   Same workflow as full GDD-ENS training (this is usually submitted as a bash job array so all 10 submit to HPC and are ran at the same time, otherwise needs to be run ten separate times for generation of the ten independent models)

    ```{linux train_ens_classifier}
    python modified_scripts/split_data_orig.py train_filename test_filename
    python modified_scripts/train_gdd_nn_orig.py fold
    ```
6.  Generate outputs for ensemble model on GDD-RF dataset
    -   same workflow as full GDD-ENS training - output is **ensemble_results_orig.csv** datatable of all results

    ```{linux test_ensemble}
    python modified_scripts/gdd_ensemble_orig.py 
    ```

#### Second comparison: GDD-RF architecture on GDD-ENS train and test sets

7.  Train random forest on GDD-ENS feature table
    -   **feature_table_filename** - training data table (ex 'comparison_data/ft_train_labelled_dedup.csv')
    -   **model_filename** - filepath where model is stored (ex 'gdd_rf.rda')
    -   optional: **-t** indicates test classifier with limited training regime used during dev.

    ```{linux train_rf_classifier}
    Rscript modified_scripts/train_classifier_mod.R feature_table_filename model_filename
    ```

8.  Generate outputs for RF model on GDD-ENS dataset
    -   **model_filename** - filepath where model is stored (ex 'gdd_rf.rda')
    -   **test_filename** - filepath where model test set is stored (ex 'comparison_data/ft_test_labelled.csv')
    -   **output_filename** - filepath where model output will be stored (ex 'rf_results_upd.csv')

    ```{linux test_rf}
    Rscript modified_scripts/predict_new_mod.R model_filename test_filename output_filename
    ```
#### Compare outputs of both models
9. Run comparison script
- if using re-trained outputs separate from the preloaded models need to change the associated filepaths within thes script
-   **output** = final_comp_res.csv, comparison datatable summarizing results from preloaded_results folder
    ```{linux test_rf}
    python modified_scripts/compare_models.py
    ```
