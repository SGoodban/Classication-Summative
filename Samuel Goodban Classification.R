telecoms <- read.csv("telecom.csv")
str(telecoms)

## Data Visualisation
library("skimr")
library("tidyverse")
library("ggplot2")
library("GGally")
library("ggforce")

skim(telecoms)

ggplot(telecoms,
       aes(x = MonthlyCharges, y = Churn, col = InternetService)) +
  geom_point()

ggplot(telecoms,
       aes(x = InternetService, y = MonthlyCharges)) +
  geom_point()


ggpairs(telecoms |> select(tenure, Contract, Churn, MonthlyCharges, TotalCharges),
        aes(color = Churn))


telecoms.bycontract <- telecoms |> 
  group_by(Contract) |> 
  summarise(total = n(),
            Churns = sum(Churn=="Yes"),
            pct.Churned = Churns/total*100)
telecoms.bycontract



telecoms.par <- telecoms |>
  select(PhoneService, Churn, MultipleLines, InternetService, TechSupport) |>
  group_by(PhoneService, Churn, MultipleLines, InternetService, TechSupport) |>
  summarize(value = n())

ggplot(telecoms.par |> gather_set_data(x = c(1, 3:5)),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor(Churn)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()


DataExplorer::plot_bar(telecoms, ncol = 3)
DataExplorer::plot_histogram(telecoms, ncol = 3)
DataExplorer::plot_boxplot(telecoms, by = "Churn", ncol = 3)

##Data modifications
telecoms <- telecoms |>
  mutate(Partner = ifelse(Partner=="No", 0, 1)) |>
  mutate(Dependents = ifelse(Dependents=="No", 0, 1)) |>
  mutate(Family = case_when(
    Partner + Dependents > 0 ~ "Yes",
    TRUE ~ "No"
  )) |>
  select(-Partner, -Dependents)

telecoms$Family <- factor(telecoms$Family)

telecoms2 <- telecoms |>
  select(-TotalCharges)



## Models

#training + test split
library("rsample")
set.seed(212)
# by setting the seed we know everyone will see the same results

tele_split <- initial_split(telecoms, prop = 3/4)
tele_train <- training(tele_split)
tele_test <- testing(tele_split)
tele2_split <- initial_split(telecoms2, prop = 3/4)
tele2_train <- training(tele_split)
tele2_test <- testing(tele_split)

#Logistic Regression
fit.lr <- glm(as.factor(Churn) ~ ., binomial, tele2_train)
levels(telecoms$Churn)
summary(fit.lr)

pred.lr <- predict(fit.lr, na.omit(tele2_test), type = "response")

png(file = "./LR_pred_hist.png")
ggplot(data.frame(x = pred.lr), aes(x = x)) + geom_histogram()
dev.off()

conf.mat <- table(`true churn` = na.omit(tele_test)$Churn, `predict churn` = ifelse(pred.lr > 0.5, 'Yes', 'No'))
conf.mat
conf.mat/rowSums(conf.mat)*100


###MLR3 Set up
library("mlr3")
library("mlr3learners")
library("data.table")
library("mlr3verse")
task_telecoms <- TaskClassif$new(id = "telecoms",
                                 backend = telecoms, 
                                 target = "Churn",
                                 positive = "Yes")
task_telecoms2 <- TaskClassif$new(id = "telecoms2",
                                  backend = telecoms2, 
                                  target = "Churn",
                                  positive = "Yes")
cv5 <- rsmp("cv", folds = 5)
#boot <- rsmp("bootstrap")
#boot$instantiate(task_telecoms2)
cv5$instantiate(task_telecoms)
cv5$instantiate(task_telecoms2)
#could do boostrap as well

#dealing with missingness
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")


##Baseline
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_baseline$train(task_telecoms)
pred <- lrn_baseline$predict(task_telecoms)
pred$score(list(msr("classif.ce"),
                msr("classif.acc"),
                msr("classif.auc"),
                msr("classif.fpr"),
                msr("classif.fnr")))
pred$confusion




##MLR3 Logistic
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
#apply imputation for missing values:
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)


##CART
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.033)
#best cp found later


##Random Forrest
#Parameter Selection
num.trees_l = c(5,10,15,20,25,30)
max_depth_l = c(5,10,15,20,25,30)

current = 0
for (i in num.trees_l){
  for (j in max_depth_l){
    lrn_forrest <- lrn("classif.ranger", predict_type = "prob", num.trees = i, max.depth= j)
    pl_forrest <- pl_missing %>>%
      po(lrn_forrest)
    res <- resample(task_telecoms, pl_forrest, cv5, store_models = FALSE)
    if (res$aggregate()>current){
      current = res$aggregate()
      best = c(i,j)
    }
  }
}
best

#Model
#Parameter selection yielded 5, 25
lrn_forrest <- lrn("classif.ranger", predict_type = "prob", num.trees = 5, max.depth= 25)
pl_forrest <- pl_missing %>>%
  po(lrn_forrest)


##Xg Boost
#Parameter Selection
nrounds_l = c(5,10,15,20,25,30)
max_depth_l = c(5,10,15,20,25,30)

current = 0
for (i in nrounds_l){
  for (j in max_depth_l){
    lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob", nrounds = i, max_depth= j)
    pl_xgb <- po("encode") %>>%
      po(lrn_xgboost)
    res <- resample(task_telecoms, pl_xgb, cv5, store_models = TRUE)
    if (res$aggregate()>current){
      current = res$aggregate()
      best = c(i,j)
    }
  }
}
best

#Model
#Pipeline which encodes and then fits an XGBoost model
#Parameter selection yielded 15, 15
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob", nrounds = 15, max_depth= 15)
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

###Output
res <- benchmark(data.table(
  task       = list(task_telecoms,
                    task_telecoms,
                    task_telecoms,
                    task_telecoms,
                    task_telecoms,
                    task_telecoms2,
                    task_telecoms),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cv,
                    lrn_cart_cp,
                    pl_xgb,
                    pl_log_reg,
                    pl_forrest),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"),
                   msr("classif.auc")))


##Tree plots
# get the trees (2nd model fitted), by asking for second set of resample
# results
trees <- res$resample_result(2)

# look at the tree from first CV iteration:
tree1 <- trees$learners[[1]]

# This is a fitted rpart object, so we can look at the model within
tree1_rpart <- tree1$model

# Plotting
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

# Enable nested cross validation
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

#pruned tree
res_cart_cv <- resample(task_telecoms, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)


###Neural Network
#Split data into training, test and validation
# First get the training
tele_split <- initial_split(telecoms2)
tele_train <- training(tele_split)
# Then further split the training into validate and test
tele_split2 <- initial_split(testing(tele_split), 0.5)
tele_validate <- training(tele_split2)
tele_test <- testing(tele_split2)


library("recipes")

cake <- recipe(Churn ~ ., data = telecoms2) %>%
  step_impute_mean(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = tele_train) # learn all the parameters of preprocessing on the training data

tele_train_final <- bake(cake, new_data = tele_train) # apply preprocessing to training data
tele_validate_final <- bake(cake, new_data = tele_validate) # apply preprocessing to validation data
tele_test_final <- bake(cake, new_data = tele_test) # apply preprocessing to testing data



library("keras")

tele_train_x <- tele_train_final %>%
  select(-starts_with("Churn_")) %>%
  as.matrix()
tele_train_y <- tele_train_final %>%
  select(Churn_No) %>%
  as.matrix()

tele_validate_x <- tele_validate_final %>%
  select(-starts_with("Churn_")) %>%
  as.matrix()
tele_validate_y <- tele_validate_final %>%
  select(Churn_No) %>%
  as.matrix()

tele_test_x <- tele_test_final %>%
  select(-starts_with("Churn_")) %>%
  as.matrix()
tele_test_y <- tele_test_final %>%
  select(Churn_No) %>%
  as.matrix()

# We make 4 neural network with 4 hidden layers, 32 neurons in each
# and an output to a binary classification
#1
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(tele_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  tele_train_x, tele_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(tele_validate_x, tele_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict(tele_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict(tele_test_x) %>% `>`(0.5) %>% as.integer()

table(pred_test_res, tele_test_y)
yardstick::accuracy_vec(as.factor(tele_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(tele_test_y, levels = c("1","0")),
                       c(pred_test_prob))

#2
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(tele_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  tele_train_x, tele_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(tele_validate_x, tele_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict(tele_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict(tele_test_x) %>% `>`(0.5) %>% as.integer()

table(pred_test_res, tele_test_y)
yardstick::accuracy_vec(as.factor(tele_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(tele_test_y, levels = c("1","0")),
                       c(pred_test_prob))


#3
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "sigmoid",
              input_shape = c(ncol(tele_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  tele_train_x, tele_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(tele_validate_x, tele_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict(tele_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict(tele_test_x) %>% `>`(0.5) %>% as.integer()

table(pred_test_res, tele_test_y)
yardstick::accuracy_vec(as.factor(tele_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(tele_test_y, levels = c("1","0")),
                       c(pred_test_prob))


#4
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "sigmoid",
              input_shape = c(ncol(tele_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  tele_train_x, tele_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(tele_validate_x, tele_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict(tele_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict(tele_test_x) %>% `>`(0.5) %>% as.integer()

table(pred_test_res, tele_test_y)
yardstick::accuracy_vec(as.factor(tele_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(tele_test_y, levels = c("1","0")),
                       c(pred_test_prob))
