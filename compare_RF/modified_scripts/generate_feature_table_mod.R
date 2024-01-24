#edited from script written by Youyun Zheng, Niedzica Camacho, Alex Penson
#all library files - install if not loaded

#install.packages(c('MLmetrics', 'binom', 'covr', 'openxlsx', 'cran'), repos='http://cran.us.r-project.org')
#install.packages("rfFC", repos="http://R-Forge.R-project.org")
#install.packages('MolecularDiagnosis_1.2.0.tar.gz',repos = NULL, type="source")

library(data.table)
library(caret)
library(randomForest)
library(glmnet)
library(MLmetrics)
library(binom)
library(covr)
library(openxlsx)
library(rfFC)
library(MolecularDiagnosis)


source("modified_scripts/file_functions.R")
source("modified_scripts/feature_categories.R")

.generate_feature_table <- function(repository_folder){

  data_clinical_sample <- fread(
    file.path(repository_folder, "data_clinical_sample.txt"),
    skip = 4)
  data_clinical_patient <- fread(
    file.path(repository_folder, "data_clinical_patient.txt"),
    skip = 4)
  data_clinical_sample <- merge(
    data_clinical_sample,
    data_clinical_patient,
    by = "PATIENT_ID",
    all.x = T)

  ### get supplementary info
  #supplement_file_tmp <- tempfile()
  #download.file(
    #"https://media.nature.com/original/nature-assets/nm/journal/v23/n6/extref/nm.4333-S2.xlsx",
    #destfile = supplement_file_tmp,
    #quiet = TRUE
  #)
  #file is no longer stored at this link - requires moving the included file '
  data_clinical_sample_extra <- as.data.table(readxl::read_xlsx(file.path(repository_folder, '41591_2017_BFnm4333_MOESM22_ESM.xlsx'), sheet = 1, skip = 3))
  #file.remove(supplement_file_tmp)


  data_clinical_sample <- merge(
    data_clinical_sample,
    data_clinical_sample_extra[, .(
      SAMPLE_ID = Assay_ID,
      CANCER_TYPE = GeneralTumorType,
      CANCER_TYPE_DETAILED = DetailedTumorType)],
    by = "SAMPLE_ID",
    all.x = T
  )

  # cancer_type_key <- file.path(code_dir, "tumor_types.txt")
  # # if(cancer_type_key == "none"){
  # #   input_cancertypes <- NULL
  # # } else {
  # input_cancertypes <- suppressWarnings(fread(cancer_type_key))
  # input_cancertypes <- input_cancertypes[, .(CANCER_TYPE,
  #                                            CANCER_TYPE_DETAILED,
  #                                            Cancer_Type)]
  # input_cancertypes <- input_cancertypes[Cancer_Type != ""]
  # input_cancertypes <- input_cancertypes[!is.na(Cancer_Type)]
  # input_cancertypes[, Cancer_Type := make.names(Cancer_Type)]
  # # }
  # # setnames(input_cancertypes, "Cancer_Type", "iCancer_Type")

  feature_table <- merge(
    data_clinical_sample,
    MolecularDiagnosis::cancertypes[, .(CANCER_TYPE,
                                        CANCER_TYPE_DETAILED,
                                        Cancer_Type)],
    by = c("CANCER_TYPE",
           "CANCER_TYPE_DETAILED"),
    all.x = TRUE)
  feature_table[, Gender_F := as.integer(SEX == "Female")]
  feature_table[is.na(Cancer_Type), Cancer_Type := "other"]
  feature_table[, Classification_Category := "train"]
  feature_table[Cancer_Type == "other", Classification_Category := "other"]
  feature_table <- feature_table[, .(
    SAMPLE_ID,
    CANCER_TYPE,
    CANCER_TYPE_DETAILED,
    SAMPLE_TYPE,
    PRIMARY_SITE,
    METASTATIC_SITE,
    Cancer_Type,
    Classification_Category,
    Gender_F
  )]

  sigs <- fread(file.path(repository_folder, "msk_impact_2017_data_mutations_uniprot.30sigs.txt"))
  sigsm <- melt.data.table(sigs, id.vars = c("Sample Name", "Number of Mutations"))

  sig_names <- c(
    "Signature.1" = "Age",
    "Signature.2" = "APOBEC",
    "Signature.3" = "BRCA",
    "Signature.4" = "Smoking",
    "Signature.5",
    "Signature.6" = "MMR",
    "Signature.7" = "UV",
    "Signature.8",
    "Signature.9",
    "Signature.10" = "POLE",
    "Signature.11" = "TMZ",
    "Signature.12",
    "Signature.13" = "APOBEC",
    "Signature.14",
    "Signature.15" = "MMR",
    "Signature.16",
    "Signature.17",
    "Signature.18",
    "Signature.19",
    "Signature.20" = "MMR",
    "Signature.21",
    "Signature.22",
    "Signature.23",
    "Signature.24" = "Smoking",
    "Signature.25",
    "Signature.26" = "MMR",
    "Signature.27",
    "Signature.28",
    "Signature.29",
    "Signature.30"
  )
  sigsm[, sig_name := sig_names[variable]]
  sigsm <- sigsm[!sig_name %like% "^Signature" &
                   `Number of Mutations` >= 20 &
                   value > 0.4]
  # sigsm[, variable := as.character(variable)]
  sigs <- dcast.data.table(
    sigsm,
    `Sample Name` ~ sig_name,
    value.var = "value",
    fill = 0,
    fun.aggregate = function(x)
      length(x > 0) > 0
  )
  setnames(sigs, c("SAMPLE_ID", paste0("Sig_", names(sigs)[-1])))



  input_seg <- "msk_impact_2017_data_cna_hg19.seg"
  seg <- suppressWarnings(fread(showProgress = F, file.path(repository_folder, input_seg)))
  # }

  # if(include_focal_cn_portal == TRUE) {
  input_cn <- "data_CNA.txt"
  cn <- suppressWarnings(fread(showProgress = F, file.path(repository_folder, input_cn)))

  input_SV <- "data_fusions.txt"
  SV <- suppressWarnings(fread(showProgress = F, file.path(repository_folder, input_SV)))

  maf <- suppressWarnings(
    fread(
      showProgress = F,
      file.path(repository_folder,
                "data_mutations_uniprot.txt")
    )
  )

  ## purity
  maf[, t_depth := t_alt_count + t_ref_count]
  maf[, t_var_freq := t_alt_count / t_depth]
  max_vaf <- maf[, .(max_vaf = max(t_var_freq)), keyby = .(SAMPLE_ID = Tumor_Sample_Barcode)]
  max_logr <- seg[, .(max_logr = max(abs(seg.mean[num.mark >= 100]))), keyby = .(SAMPLE_ID = ID)]
  purity_est <- merge(max_vaf, max_logr, all = T)
  feature_table <- merge(feature_table, purity_est, by = "SAMPLE_ID", all.x = T)
  feature_table[is.na(max_logr) | is.infinite(max_logr), max_logr := 0]
  feature_table[is.na(max_vaf), max_vaf := 0]
  feature_table[max_vaf < 0.1 & max_logr < 0.2, Classification_Category := "low_purity"]
  feature_table[, max_vaf := NULL]
  feature_table[, max_logr := NULL]

  mutation_count <- maf[, .(SNVCount = .SD[Variant_Type == "SNP", .N],
                            INDELCount = .SD[Variant_Type %in% c("INS", "DEL"), .N]),
                        keyby = .(SAMPLE_ID = Tumor_Sample_Barcode)]

  feature_table <- merge(feature_table, mutation_count, by = "SAMPLE_ID", all.x = T)

  feature_table[is.na(SNVCount), SNVCount := 0]
  feature_table[is.na(INDELCount), INDELCount := 0]

  feature_table[, SEQ_ASSAY_ID := stringr::str_extract(
    SAMPLE_ID,
    "(?<=[P]-[0-9]{7,12}-T[0-9]{2}-)[A-Z0-9]{3}"),
    SAMPLE_ID]

  targeted_Mb <- c(
    "IM3" = 1048641,
    "IM5" = 1206011,
    "IM6" = 1699692
  ) / 1e6
  feature_table[, SNVCount := SNVCount / targeted_Mb[SEQ_ASSAY_ID]]
  feature_table[, INDELCount := INDELCount / targeted_Mb[SEQ_ASSAY_ID]]

  feature_table[, LogSNV_Mb := log10(SNVCount+1)]
  feature_table[, LogINDEL_Mb := log10(INDELCount+1)]
  feature_table[, SNVCount := NULL]
  feature_table[, INDELCount := NULL]
  feature_table[, SEQ_ASSAY_ID := NULL]



  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(mutations(maf = maf)))
  #if(include_truncating_mutations == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(truncating_mutations(maf = maf)))
  #if(include_hotspots == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(hotspots(maf = maf)))
  #if(include_focal_cn_portal == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(focal_cn_portal(cn = cn)))
  #if(include_broad_cn == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(broad_cn(seg = seg)))
  #if(include_mutational_signatures == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, sigs)
  #if(include_structural_variants == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(fusions(SV = SV)))
  #if(include_gender == TRUE)
  # feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(gender(clinical = clinical)))
  #if(include_cn_burden == TRUE)
  feature_table <- merge(all.x = TRUE, by="SAMPLE_ID", feature_table, as.data.table(cn_burden(seg = seg)))

  feature_table[is.na(feature_table)] <- 0

  feature_table[, SEQ_ASSAY_ID := stringr::str_extract(
    SAMPLE_ID,
    "(?<=[P]-[0-9]{7,12}-T[0-9]{2}-)[A-Z0-9]{3}"),
    SAMPLE_ID]
  feature_table[, PATIENT_ID := stringr::str_extract(
    SAMPLE_ID,
    "^P-[0-9]{7,12}"),
    SAMPLE_ID]
  feature_table <- feature_table[order(SAMPLE_TYPE, -SEQ_ASSAY_ID, SAMPLE_ID)]
  duplicate_ids <- feature_table[Classification_Category == "train"][duplicated(PATIENT_ID)][, SAMPLE_ID]
  feature_table[SAMPLE_ID %in% duplicate_ids, Classification_Category := "secondary"]
  feature_table[, PATIENT_ID := NULL]
  feature_table[, SEQ_ASSAY_ID := NULL]

  feature_table
}

#' generate_feature_table
#'
#' @param repository_folder
#'
#' @return feature_table
#' @export
#'
generate_feature_table <- function(arg_line = NA){
  ### process args

  ##MODIFIED - directly load R arg line
  #if(!is.na(arg_line)) {
    # reading from arg_line
   # raw_args <- unlist(stringr::str_split(arg_line, " "))
 # } else {
    # batch
    #raw_args <- commandArgs(TRUE)
  #}

  raw_args <- arg_line
  repository_folder <- raw_args[[1]]; raw_args <- raw_args[-1]
  feature_table_filename <- raw_args[[1]]; raw_args <- raw_args[-1]

  feature_table <- .generate_feature_table(repository_folder = repository_folder)
  write.tab(feature_table, feature_table_filename)
}

arg_line <- commandArgs(trailingOnly=TRUE)
generate_feature_table(arg_line)
