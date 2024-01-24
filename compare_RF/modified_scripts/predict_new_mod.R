#edited from script written by Youyun Zheng, Niedzica Camacho, Alex Penson

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

source('modified_scripts/GDDmetrics.R')
source('modified_scripts/file_functions.R')

predict_new <- function(arg_line = NA){
  #MODIFIED FROM ORIGINAL
  ### process args
  #if(!is.na(arg_line) | interactive()) {
    # print("reading from arg_line")
   # raw_args <- unlist(stringr::str_split(arg_line, " "))
  #} else {
    # print("batch")
  #  raw_args <- commandArgs(TRUE)
  #}
  # print(raw_args)
  raw_args <- arg_line
  option_list <- list(
    optparse::make_option(c("-t", "--test"),
                          default = FALSE,
                          action="store_true",
                          help="train a limited test classifier")
  )
  if(any(sapply(
    option_list,
    function(option){
      option@short_flag == "-g"
    }))){
    stop("cannot use short option '-g', conflicts with Rscript --gui")
  }

  opts <- optparse::parse_args(
    optparse::OptionParser(option_list=option_list),
    args = raw_args,
    positional_arguments = TRUE
  )

  model_filename <- opts$args[[1]]; opts$args <- opts$args[-1]
  feature_table_filename <- opts$args[[1]]; opts$args <- opts$args[-1]
  prediction_table_filename <- opts$args[[1]]; opts$args <- opts$args[-1]

  #ADDED IN
  rfe_model <- readRDS(model_filename)
  feature_table <- fread(feature_table_filename)
  Cancer_Types <- rfe_model$obsLevels
  vars <- rfe_model$optVariables

  x <- feature_table[, vars, with = F]
  rownames(x) <- feature_table$SAMPLE_ID
  model_function_list = MolecularDiagnosis::rfcal
  pred_mat <- as.data.table(matrix(nrow = nrow(x), model_function_list$prob(modelFit = rfe_model$fit$finalModel, newdata = x)))
  names(pred_mat) <- Cancer_Types
  N_top_cancer_types <- 3
  pred <- cbind(feature_table[, .(SAMPLE_ID, Classification_Category, Cancer_Type)],
                pred_mat)
  pred[, paste0("Pred", 1:N_top_cancer_types) := as.list(
    Cancer_Types[head(n=N_top_cancer_types,
                      order(.SD,
                            decreasing = T))]),
    SAMPLE_ID,
    .SDcols = Cancer_Types]
  pred[, paste0("Conf", 1:N_top_cancer_types) := as.list(round(
    digits = 4,
    as.numeric(
      head(n=N_top_cancer_types,
           sort(.SD, decreasing = T))))),
    SAMPLE_ID,
    .SDcols = Cancer_Types]

  write.tab(pred,prediction_table_filename)
}

arg_line <- commandArgs(trailingOnly=TRUE)
predict_new(arg_line)
