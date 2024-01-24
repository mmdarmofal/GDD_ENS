write.tab <- function (...){
  write.table(..., quote = F, col.names = T, row.names = F,
              sep = "\t")
}

rename_types <- function(Cancer_Types){
  Cancer_Type_Names <- gsub("\\.", " ", Cancer_Types)
  Cancer_Type_Names <- gsub(" Cancer$", "", Cancer_Type_Names)
  Cancer_Type_Names <- gsub("Non Small Cell Lung", "NSCLC", Cancer_Type_Names)
  Cancer_Type_Names <- gsub("Small Cell Lung", "SCLC", Cancer_Type_Names)
  Cancer_Type_Names <- gsub("Gastrointestinal Stromal Tumor", "GIST", Cancer_Type_Names)
  Cancer_Type_Names <- gsub("Pancreatic Neuroendocrine Tumor", "PNET", Cancer_Type_Names)
  Cancer_Type_Names
}
