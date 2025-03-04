##Cross-validation Function
DNN_cross_validation <- function(directory, trait, folds, replicates){
  library(tidyverse)
  library(keras)
  library(tensorflow)
  library(tidymodels)
  
  #Load data
  files <- list.files(path = directory, pattern = "^matrix_\\d+\\.txt$", full.names = TRUE)
  df_list <- lapply(files, read_delim, delim = "\t", col_names = TRUE)
  matrix_file <- Reduce(function(x, y) merge(x, y, by = "Environment"), df_list)
  significance_rankings <- read_delim(paste0(directory, "/", "Feature_significance_rankings.txt"))
  
  #Set percentages for permutation analysis
  percentages <- c(0.01, seq(0.05, 1, by = 0.05))
  
  #Trim data depending on trait
  matrix_file <- matrix_file %>%
    filter(!is.na(!!sym(trait)))
  
  if(trait == "meanHD") {
    matrix_file <- matrix_file %>%
      select(-matches("7[2-9]|8[0-9]|9[0-9]|10[0-9]|11[0-9]|12[0-9]|13[0-9]"))
  }
  
  #Cross-validation set up
  rows_per_fold <- ceiling(nrow(matrix_file) / folds)
  matrix_file$set <- rep(1:folds, each = rows_per_fold, length.out = nrow(matrix_file))
  
  #Initialize storage for replication results
  replication_df <- c()
  
  for(r in 1:replicates) {
    randomized_df <- matrix_file %>%
      mutate(randomized_set = sample(set))
    
    #Initialize storage for fold results
    fold_df <- c()
    
    for(i in 1:folds) {
      #Split environments into training and testing
      columns_to_remove <- c("meanGY", "meanPC", "meanPH", "meanHD", "set", "randomized_set")
      columns_to_remove <- setdiff(columns_to_remove, trait)
      
      training_dataset_with_environment <- randomized_df %>%
        filter(randomized_set != i) %>%
        select(-all_of(columns_to_remove))
      testing_dataset_with_environment <- randomized_df %>%
        filter(randomized_set == i) %>%
        select(-all_of(columns_to_remove))
      
      training_dataset <- training_dataset_with_environment %>%
        select(-Environment)
      testing_dataset <- testing_dataset_with_environment %>%
        select(-Environment)
      
      #Divide training and testing sets into input features and labels
      training_features <- training_dataset %>% dplyr::select(-!!sym(trait))
      testing_features <- testing_dataset %>% dplyr::select(-!!sym(trait))
      training_labels <- training_dataset %>% dplyr::select(sym(trait))
      testing_labels <- testing_dataset %>% dplyr::select(sym(trait))
      
      #Normalize the training and testing features
      training_mean <- apply(training_features, 2, mean)
      training_sd <- apply(training_features,2, sd)
      training_features <- as.data.frame(scale(training_features, center = training_mean, scale = training_sd))
      testing_features <- as.data.frame(scale(testing_features, center = training_mean, scale = training_sd))
      
      #Define model
      if(trait == "meanPC") {
        model <- keras_model_sequential() %>%
          layer_dense(64, activation = 'relu', input_shape = ncol(train_features), kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.2) %>%
          layer_dense(64, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.2) %>%
          layer_dense(units = 1)
        
        model %>% compile(
          optimizer = optimizer_adam(learning_rate = 0.001),
          loss = "mse",
          metrics = c("mae"))
        
        model %>% fit(
          as.matrix(train_features),
          as.matrix(train_labels),
          epochs = 75,
          verbose = 0,
          validation_split = 0)
      } else {
        model <- keras_model_sequential() %>%
          layer_dense(512, activation = 'relu', input_shape = ncol(training_features), kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(256, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(256, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(128, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(128, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
          layer_batch_normalization() %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(units = 1)
        
        model %>% compile(
          optimizer = optimizer_adam(learning_rate = 0.001),
          loss = "mse",
          metrics = c("mae"))
        
        model %>% fit(
          as.matrix(training_features),
          as.matrix(training_labels),
          epochs = 150,
          verbose = 0,
          validation_split = 0) 
      }
      
      #Initialize storage for the permutation results
      permutation_results_df <- c()
      
      for(p in percentages) {
        #Specify the ranking column based on the trait
        ranking_column <- gsub("mean", "", trait) %>% paste0("_rank") 
        lower_threshold <- quantile(significance_rankings[[ranking_column]], p)
        upper_threshold <- quantile(significance_rankings[[ranking_column]], (1-p))
        
        #Create sub sets corresponding to the top and bottom of the significance rankings as well as a random set
        top_df <- significance_rankings %>%
          filter(get(ranking_column) < lower_threshold)
        bottom_df <- significance_rankings %>%
          filter(get(ranking_column) > upper_threshold)
        random_df <- significance_rankings %>%
          slice_sample(prop = p)
        
        #Permute the selected features
        matching_columns_top <- names(randomized_df) %in% top_df$Feature
        shuffled_matching_columns_top <- randomized_df[, matching_columns_top]
        shuffled_matching_columns_top[] <- lapply(shuffled_matching_columns_top, sample)
        shuffled_top_df <- cbind(randomized_df[, !matching_columns_top, drop = FALSE], shuffled_matching_columns_top)
        
        matching_columns_bottom <- names(randomized_df) %in% bottom_df$Feature
        shuffled_matching_columns_bottom <- randomized_df[, matching_columns_bottom]
        shuffled_matching_columns_bottom[] <- lapply(shuffled_matching_columns_bottom, sample)
        shuffled_bottom_df <- cbind(randomized_df[, !matching_columns_bottom, drop = FALSE], shuffled_matching_columns_bottom)
        
        matching_columns_random <- names(randomized_df) %in% random_df$Feature
        shuffled_matching_columns_random <- randomized_df[, matching_columns_random]
        shuffled_matching_columns_random[] <- lapply(shuffled_matching_columns_random, sample)
        shuffled_random_df <- cbind(randomized_df[, !matching_columns_random, drop = FALSE], shuffled_matching_columns_random)
        
        #Correct feature order in shuffled data frames to match unaltered data, retain only the testing environments, and remove unneeded columns
        desired_column_order <- names(randomized_df)
        
        shuffled_top_df_ordered <- shuffled_top_df %>%
          select(desired_column_order) %>%
          filter(randomized_set == i) %>%
          select(-all_of(columns_to_remove)) %>%
          select(-Environment)
        shuffled_bottom_df_ordered <- shuffled_bottom_df %>%
          select(desired_column_order) %>%
          filter(randomized_set == i) %>%
          select(-all_of(columns_to_remove)) %>%
          select(-Environment)
        shuffled_random_df_ordered <- shuffled_random_df %>%
          select(desired_column_order) %>%
          filter(randomized_set == i) %>%
          select(-all_of(columns_to_remove)) %>%
          select(-Environment)
        
        #Separate permuted data into features and labels
        test_features_top <- shuffled_top_df_ordered %>% dplyr::select(-!!sym(trait))
        test_features_bottom <- shuffled_bottom_df_ordered %>% dplyr::select(-!!sym(trait))
        test_features_random <- shuffled_random_df_ordered %>% dplyr::select(-!!sym(trait))
        
        test_labels_top <- shuffled_top_df_ordered %>% dplyr::select(!!sym(trait))
        test_labels_bottom <- shuffled_bottom_df_ordered %>% dplyr::select(!!sym(trait))
        test_labels_random <- shuffled_random_df_ordered %>% dplyr::select(!!sym(trait))
        
        #Normalize permuted data
        test_features_top <- as.data.frame(scale(test_features_top, center = training_mean, scale = training_sd))
        test_features_bottom <- as.data.frame(scale(test_features_bottom, center = training_mean, scale = training_sd))
        test_features_random <- as.data.frame(scale(test_features_random, center = training_mean, scale = training_sd))
        
        #Generate predictions for each set of testing features
        unaltered_predictions <- predict(model, as.matrix(testing_features))
        top_permuted_predictions <- predict(model, as.matrix(test_features_top)) 
        bottom_permuted_predictions <- predict(model, as.matrix(test_features_bottom))
        random_permuted_predictions <- predict(model, as.matrix(test_features_random))
        
        #Combine predictions into data frame
        combined_df <- data.frame(Environment = testing_dataset_with_environment$Environment, Observed_value = testing_dataset[[trait]], Predicted_value_unaltered = as.numeric(unaltered_predictions), Predicted_value_top_permuted = as.numeric(top_permuted_predictions), Predicted_value_bottom_permuted = as.numeric(bottom_permuted_predictions), Predicted_value_random_permuted = as.numeric(random_permuted_predictions))
        permutation_results_df[[length(permutation_results_df) + 1]] <- combined_df %>%
          mutate(Percentage_permuted = p)
      }
      
      #Combine all the permuted results into one data frame for the fold
      permutations_combined_df <- do.call(rbind, permutation_results_df)
      fold_df[[i]] <- permutations_combined_df %>%
        mutate(Fold = i)
    }
    
    #Combine all the fold results into one data frame for the replicate
    folds_combined_df <- do.call(rbind, fold_df)
    replication_df[[r]] <- folds_combined_df %>%
      mutate(Replicate = r)
  }
  
  #Combine all the replication results into one data frame
  final_df <- do.call(rbind, replication_df)
  
  #Save the results
  write_delim(final_df, paste0(directory, "/", "Cross_validation_results_", trait, ".txt"), delim = "\t", col_names = TRUE)
}


##Forecasting function
DNN_forecasting <- function(directory, trait, testing_years) {
  library(tidyverse)
  library(keras)
  library(tensorflow)
  library(tidymodels)
  
  #Load data
  files <- list.files(path = directory, pattern = "^matrix_\\d+\\.txt$", full.names = TRUE)
  df_list <- lapply(files, read_delim, delim = "\t", col_names = TRUE)
  matrix_file <- Reduce(function(x, y) merge(x, y, by = "Environment"), df_list)
  significance_rankings <- read_delim(paste0(directory, "/", "Feature_significance_rankings.txt"))
  
  #Set percentages for permutation analysis
  percentages <- c(0.01, seq(0.05, 1, by = 0.05))
  
  #Trim data depending on trait
  matrix_file <- matrix_file %>%
    filter(!is.na(!!sym(trait)))
  
  if(trait == "meanHD") {
    matrix_file <- matrix_file %>%
      select(-matches("7[2-9]|8[0-9]|9[0-9]|10[0-9]|11[0-9]|12[0-9]|13[0-9]"))
  }
  
  #Add year column
  matrix_file <- matrix_file %>%
    mutate(Year = sub(".*_", "", Environment)) %>%
    mutate(Year = as.numeric(Year)) %>%
    mutate(Year = Year + 2000)
  
  #Initialize storage for year results
  year_df <- c()
  
  for(t in testing_years) {
    #Split environments into training and testing
    columns_to_remove <- c("meanGY", "meanPC", "meanPH", "meanHD")
    columns_to_remove <- setdiff(columns_to_remove, trait)
    
    training_dataset_with_environment <- matrix_file %>%
      filter(Year < t) %>%
      select(-all_of(columns_to_remove))
    testing_dataset_with_environment <- matrix_file %>%
      filter(Year == t) %>%
      select(-all_of(columns_to_remove))
    
    training_dataset <- training_dataset_with_environment %>%
      select(-Environment) %>%
      select(-Year)
    testing_dataset <- testing_dataset_with_environment %>%
      select(-Environment) %>%
      select(-Year)
    
    #Divide training and testing sets into input features and labels
    training_features <- training_dataset %>% dplyr::select(-!!sym(trait))
    testing_features <- testing_dataset %>% dplyr::select(-!!sym(trait))
    training_labels <- training_dataset %>% dplyr::select(sym(trait))
    testing_labels <- testing_dataset %>% dplyr::select(sym(trait))
    
    #Normalize the training and testing features
    training_mean <- apply(training_features, 2, mean)
    training_sd <- apply(training_features,2, sd)
    training_features <- as.data.frame(scale(training_features, center = training_mean, scale = training_sd))
    testing_features <- as.data.frame(scale(testing_features, center = training_mean, scale = training_sd))
    
    #Define model
    if(trait == "meanPC") {
      model <- keras_model_sequential() %>%
        layer_dense(64, activation = 'relu', input_shape = ncol(train_features), kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(64, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.2) %>%
        layer_dense(units = 1)
      
      model %>% compile(
        optimizer = optimizer_adam(learning_rate = 0.001),
        loss = "mse",
        metrics = c("mae"))
      
      model %>% fit(
        as.matrix(train_features),
        as.matrix(train_labels),
        epochs = 75,
        verbose = 0,
        validation_split = 0)
    } else {
      model <- keras_model_sequential() %>%
        layer_dense(512, activation = 'relu', input_shape = ncol(training_features), kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(256, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(256, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(128, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(128, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(units = 1)
      
      model %>% compile(
        optimizer = optimizer_adam(learning_rate = 0.001),
        loss = "mse",
        metrics = c("mae"))
      
      model %>% fit(
        as.matrix(training_features),
        as.matrix(training_labels),
        epochs = 150,
        verbose = 0,
        validation_split = 0) 
    }
    
    #Initialize storage for the permutation results
    permutation_results_df <- c()
    
    for(p in percentages) {
      #Specify the ranking column based on the trait
      ranking_column <- gsub("mean", "", trait) %>% paste0("_rank") 
      lower_threshold <- quantile(significance_rankings[[ranking_column]], p)
      upper_threshold <- quantile(significance_rankings[[ranking_column]], (1-p))
      
      #Create sub sets corresponding to the top and bottom of the significance rankings as well as a random set
      top_df <- significance_rankings %>%
        filter(get(ranking_column) < lower_threshold)
      bottom_df <- significance_rankings %>%
        filter(get(ranking_column) > upper_threshold)
      random_df <- significance_rankings %>%
        slice_sample(prop = p)
      
      #Permute the selected features
      matching_columns_top <- names(matrix_file) %in% top_df$Feature
      shuffled_matching_columns_top <- matrix_file[, matching_columns_top]
      shuffled_matching_columns_top[] <- lapply(shuffled_matching_columns_top, sample)
      shuffled_top_df <- cbind(matrix_file[, !matching_columns_top, drop = FALSE], shuffled_matching_columns_top)
      
      matching_columns_bottom <- names(matrix_file) %in% bottom_df$Feature
      shuffled_matching_columns_bottom <- matrix_file[, matching_columns_bottom]
      shuffled_matching_columns_bottom[] <- lapply(shuffled_matching_columns_bottom, sample)
      shuffled_bottom_df <- cbind(matrix_file[, !matching_columns_bottom, drop = FALSE], shuffled_matching_columns_bottom)
      
      matching_columns_random <- names(matrix_file) %in% random_df$Feature
      shuffled_matching_columns_random <- matrix_file[, matching_columns_random]
      shuffled_matching_columns_random[] <- lapply(shuffled_matching_columns_random, sample)
      shuffled_random_df <- cbind(matrix_file[, !matching_columns_random, drop = FALSE], shuffled_matching_columns_random)
      
      #Correct feature order in shuffled data frames to match unaltered data, retain only the testing environments, and remove unneeded columns
      desired_column_order <- names(matrix_file)
      
      shuffled_top_df_ordered <- shuffled_top_df %>%
        select(desired_column_order) %>%
        filter(Year == t) %>%
        select(-all_of(columns_to_remove)) %>%
        select(-Environment) %>%
        select(-Year)
      shuffled_bottom_df_ordered <- shuffled_bottom_df %>%
        select(desired_column_order) %>%
        filter(Year == t) %>%
        select(-all_of(columns_to_remove)) %>%
        select(-Environment) %>%
        select(-Year)
      shuffled_random_df_ordered <- shuffled_random_df %>%
        select(desired_column_order) %>%
        filter(Year == t) %>%
        select(-all_of(columns_to_remove)) %>%
        select(-Environment) %>%
        select(-Year)
      
      #Separate permuted data into features and labels
      test_features_top <- shuffled_top_df_ordered %>% dplyr::select(-!!sym(trait))
      test_features_bottom <- shuffled_bottom_df_ordered %>% dplyr::select(-!!sym(trait))
      test_features_random <- shuffled_random_df_ordered %>% dplyr::select(-!!sym(trait))
      
      test_labels_top <- shuffled_top_df_ordered %>% dplyr::select(!!sym(trait))
      test_labels_bottom <- shuffled_bottom_df_ordered %>% dplyr::select(!!sym(trait))
      test_labels_random <- shuffled_random_df_ordered %>% dplyr::select(!!sym(trait))
      
      #Normalize permuted data
      test_features_top <- as.data.frame(scale(test_features_top, center = training_mean, scale = training_sd))
      test_features_bottom <- as.data.frame(scale(test_features_bottom, center = training_mean, scale = training_sd))
      test_features_random <- as.data.frame(scale(test_features_random, center = training_mean, scale = training_sd))
      
      #Generate predictions for each set of testing features
      unaltered_predictions <- predict(model, as.matrix(testing_features))
      top_permuted_predictions <- predict(model, as.matrix(test_features_top)) 
      bottom_permuted_predictions <- predict(model, as.matrix(test_features_bottom))
      random_permuted_predictions <- predict(model, as.matrix(test_features_random))
      
      #Combine predictions into data frame
      combined_df <- data.frame(Environment = testing_dataset_with_environment$Environment, Observed_value = testing_dataset[[trait]], Predicted_value_unaltered = as.numeric(unaltered_predictions), Predicted_value_top_permuted = as.numeric(top_permuted_predictions), Predicted_value_bottom_permuted = as.numeric(bottom_permuted_predictions), Predicted_value_random_permuted = as.numeric(random_permuted_predictions))
      permutation_results_df[[length(permutation_results_df) + 1]] <- combined_df %>%
        mutate(Percentage_permuted = p)
    }
    
    #Combine all the permuted results into one data frame for the year
    permutations_combined_df <- do.call(rbind, permutation_results_df)
    year_df[[t]] <- permutations_combined_df %>%
      mutate(Testing_year = t)
  }
  
  #Combine all the year results into one data frame
  final_df <- do.call(rbind, year_df)
  
  #Save the results
  write_delim(final_df, paste0(directory, "/", "Forecasting_results_", trait, ".txt"), delim = "\t", col_names = TRUE)
}






















