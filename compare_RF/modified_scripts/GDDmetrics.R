#' Title
#'
#' @param data
#' @param lev
#' @param model
#'
#' @return
GDDmetrics <- function (data, lev = NULL, model = NULL){
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("levels of observed and predicted data do not match")
  has_class_probs <- all(lev %in% colnames(data))
  if (has_class_probs) {
    lloss <- caret::mnLogLoss(data = data,
                       lev = lev,
                       model = model)

    prob_stats <- lapply(
      levels(data[, "pred"]),
      function(x) {
        obs <- ifelse(data[, "obs"] == x, 1, 0)
        prob <- data[, x]
        roc_auc <- try(ModelMetrics::auc(obs, data[, x]),
                       silent = TRUE)
        pr_auc <- try(MLmetrics::PRAUC(y_pred = data[, x],
                                       y_true = obs), silent = TRUE)
        # brier_score <- try(DescTools::BrierScore(obs, data[, x]),
        #                    silent = TRUE)
        if (inherits(pr_auc, "try-error"))
          pr_auc <- NA
        res <- c(ROC = roc_auc,
                 AUC = pr_auc
                 #,Brier = brier_score
                 )
        return(res)
      })
    # print(head(prob_stats))
    prob_stats <- do.call("rbind", prob_stats)
    prob_stats <- colMeans(prob_stats, na.rm = TRUE)
  }
  CM <-
    caret::confusionMatrix(data[, "pred"], data[, "obs"], mode = "everything")
  if (length(levels(data[, "pred"])) == 2) {
    class_stats <- CM$byClass
  }
  else {
    class_stats <- colMeans(CM$byClass)
    names(class_stats) <- paste("Mean", names(class_stats))
  }
  overall_stats <- if (has_class_probs)
    c(
      CM$overall,
      logLoss = as.numeric(lloss),
      AUC = unname(prob_stats["ROC"]),
      prAUC = unname(prob_stats["AUC"])
      #,Brier = unname(prob_stats["Brier"])
    )
  else
    CM$overall
  stats <- c(overall_stats, class_stats)
  stats <-
    stats[!names(stats) %in% c(
      "AccuracyNull",
      "AccuracyLower",
      "AccuracyUpper",
      "AccuracyPValue",
      "McnemarPValue",
      "Mean Prevalence",
      "Mean Detection Prevalence"
    )]
  names(stats) <- gsub("[[:blank:]]+", "_", names(stats))
  stat_list <-
    c(
      "Accuracy",
      "Kappa",
      "Mean_F1",
      "Mean_Sensitivity",
      "Mean_Specificity",
      "Mean_Pos_Pred_Value",
      "Mean_Neg_Pred_Value",
      "Mean_Precision",
      "Mean_Recall",
      "Mean_Detection_Rate",
      "Mean_Balanced_Accuracy"
    )
  if (has_class_probs)
    stat_list <- c("logLoss", "AUC", "prAUC",
                   # "Brier",
                   stat_list)
  if (length(levels(data[, "pred"])) == 2)
    stat_list <- gsub("^Mean_", "", stat_list)
  stats <- stats[c(stat_list)]
  return(stats)
}
