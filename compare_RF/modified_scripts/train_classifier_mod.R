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

source('modified_scripts/GDDmetrics.R')
source('modified_scripts/file_functions.R')

.train_classifier = function(
  feature_table,
  tuneGrid,
  model_function_list = MolecularDiagnosis::rfcal,
  verbose = FALSE,
  seed = 100,
  ntree = 2000,
  processes = 6,
  sizes = c(500, 600, 700, 800, 900, 1000),
  N_resamples_train = 5,
  N_resamples_rfe = 5
){

  ## https://stats.stackexchange.com/questions/214387/results-from-rfe-function-caret-to-compute-average-metrics-r/310895
  rfcalFuncs = caret::caretFuncs
  rfcalFuncs$summary = GDDmetrics
  rfcalFuncs$fit <- function (x, y, first, last, ...) train(x, y, metric = "Kappa", ...)
  rfcalFuncs$rank = function (object, x, y){
    vimp <- varImp(object, scale = FALSE)$importance
    if (!is.data.frame(vimp)){
      vimp <- as.data.frame(vimp)
    }

    if (object$modelType == "Regression"){
      vimp <- vimp[order(vimp[, 1], decreasing = TRUE), , drop = FALSE]
    } else if (all(levels(y) %in% colnames(vimp)) & !("Overall" %in% colnames(vimp))){
      avImp <- apply(vimp[, levels(y), drop = TRUE], 1, mean)
      vimp$Overall <- avImp
    }
    vimp <- vimp[order(vimp$Overall, decreasing = TRUE), , drop = FALSE]
    vimp$var <- rownames(vimp)
    vimp
  }

  # feature_table <- fread(feature_table_filename)
  feature_table <- feature_table[Classification_Category == "train"]

  y <- factor(feature_table$Cancer_Type)
  x <- feature_table[, -1:-9, with = FALSE]
  x[, c('PATIENT_ID', 'rep_drop')] <- list(NULL)

  # x <- x[,-caret::nearZeroVar(x, uniqueCut = 0.001, allowParallel = TRUE), with = F]
  # ncol(x)
  rownames(x) <- feature_table$SAMPLE_ID
  rownames(x) <- make.names(rownames(x))
  caret_processes = processes
  cl <- parallel::makeCluster(caret_processes)
  doParallel::registerDoParallel(cl)

  set.seed(seed = seed)
  N_seeds <- length(y) * 10000 ### number of possible seeds much larger than the
  N_sizes <- length(sizes) + 1
  seeds <- replicate(simplify = F,
                     N_resamples_rfe,
                     sample.int(N_seeds, size = N_sizes)
  )
  seeds <- c(seeds, sample.int(N_seeds, size = 1))
  rfe_model <- caret::rfe(
    x, y,
    ## rfe options
    sizes = sizes,
    metric = "Kappa",
    rfeControl = caret::rfeControl(
      method = "boot", number = N_resamples_rfe,
      seeds = seeds,
      functions = rfcalFuncs,
      allowParallel = TRUE
    ),

    ## train options
    method = model_function_list,
    tuneGrid = tuneGrid,
    ntree = ntree,
    trControl = caret::trainControl(
      method = "repeatedcv", number = N_resamples_train, repeats = 1,
      # method = "none",
      allowParallel = FALSE,
      savePredictions = "final",
      summaryFunction = GDDmetrics,
      #classProbs = TRUE
    )
  )

  parallel::stopCluster(cl)

  rfe_model
}
#' @import data.table
#' @import caret
#' @import randomForest
#' @import glmnet
#' @import parallel
#' @import optparse
#'
#' @name train_classifier
#' @title train_classifier
#' @description
#'
#' @param feature_table
#' @param model
#' @param tuneGrid
#' @param tuneGrid_iter
#' @param verbose
#' @param seed
#' @param processes
#'
#' @export
train_classifier <- function(arg_line = NA){
  ###MODIFIED FROM ORIGINAL
  ### process args
  #if(!is.na(arg_line) | interactive()) {
    # print("reading from arg_line")
  #  raw_args <- unlist(stringr::str_split(arg_line, " "))
  #  print(raw_args)
  #} else {
    # print("batch")
  #  raw_args <- commandArgs(TRUE)
 # }
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

  feature_table_filename <- opts$args[[1]]; opts$args <- opts$args[-1]
  model_filename <- opts$args[[1]]; opts$args <- opts$args[-1]

  feature_table <- fread(feature_table_filename)
  tuneGrid <- data.table(mtry = 10L, alpha = 0.25, lambda = 0.001, exponent = 1.2)

  if(opts$options$test == TRUE){
    ntree <- 2
    sizes <- 500
    N_resamples_train = 2
    N_resamples_rfe = 1
    processes = 1
  } else {
    ntree <- 2000
    sizes <- c(500, 600, 700, 800, 900, 1000)
    N_resamples_train = 5
    N_resamples_rfe = 5
    processes = 6
  }
  rfe_model <-
    .train_classifier(
      feature_table,
      tuneGrid,
      model_function_list = MolecularDiagnosis::rfcal,
      ntree = ntree,
      sizes = sizes,
      N_resamples_train = N_resamples_train,
      N_resamples_rfe = N_resamples_rfe,
      processes = processes
    )

  saveRDS(rfe_model, model_filename)

}



arg_line <- commandArgs(trailingOnly=TRUE)
train_classifier(arg_line)
