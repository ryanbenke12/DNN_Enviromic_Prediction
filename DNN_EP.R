##DNN-EP

#Specify the directory. Ensure that all 11 matrix files, "Source_file.R", and "Feature_significance_rankings.txt" are stored in this directory
directory <- "/your/directory"

#This source file contains the code to perform cross-validation and future forecasting.
source(paste0(directory, "/", "Source_file.R"))

#Specificy the trait of interest. Acceptible options are "meanGY", "meanHD", "meanPH", or "meanPC" for grain yield, heading date, plant height, or protein content respectively.
trait <- "meanGY"

#Specifiy the number of folds for cross-validation.
folds <- 5

#Specify how many times to replicate the analysis.
replicates = 1

#Specify the testing years for the future forecasting.
testing_years = 2019:2023

####################
##Cross-Validation##
####################
DNN_cross_validation(directory = directory, trait = trait, folds = folds, replicates = replicates)

######################
##Future Forecasting##
######################
DNN_forecasting(directory = directory, trait = trait, testing_years = testing_years)
