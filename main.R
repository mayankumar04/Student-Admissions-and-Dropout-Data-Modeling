rm(list = ls())
cat("\014")

library(tidyverse)
library(estimatr)
library(modelr)
library(caret)
library(vtable)
library(AER)
library(modelsummary)
library(MatchIt)
library(rdrobust)
library(dplyr)
library(leaps)
library(rpart)
library(rpart.plot)
library(rattle)
library(rsample)
library(ranger)
library(gbm)

load("data_frames.RData")

# Data Set 1
# Data Wrangling

student1 <- na.omit(student1)
student1 = student1 %>% mutate_if(is.character, as.factor)
set.seed(100)
n = nrow(student1)
train = sample(1:n, n * 0.7)
train.data = student1 %>% slice(train)
test.data = student1 %>% slice(-train)
train.control = trainControl(method = "cv", number = 5)
nvars = length(lm(Admission.grade ~ ., data = train.data)$coefficients) - 1

# Simple Linear Regression

lm_simple = lm(Admission.grade ~ ., data = student1)
sprintf("%.3f", rmse(lm_simple, test.data))

# Forward Leap

set.seed(100)
lm_fwd = train(Admission.grade ~ .,
  data = train.data, method = "leapForward",
  tuneGrid = data.frame(nvmax = 1:nvars),
  trControl = train.control)
sprintf("%.3f", rmse(lm_fwd, test.data))

#Backwards Leap

set.seed(100)
lm_bwd = train(Admission.grade ~ .,
  data = train.data, method = "leapBackward",
  tuneGrid = data.frame(nvmax = 1:nvars),
  trControl = train.control)
sprintf("%.3f", rmse(lm_bwd, test.data))

# Ridge Regression

set.seed(100)
lambda_seq = seq(0, 5, by = .005)
ridge = train(Admission.grade ~ ., data = train.data, 
  method = "glmnet",  preProcess = "scale", 
  trControl = train.control, 
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_seq)
)
sprintf("%.3f", rmse(ridge, test.data))

# Lasso Regression

set.seed(100)
lambda_seq = seq(0, 0.5, by = 0.0005)
lasso = train(Admission.grade ~ ., data = train.data, 
  method = "glmnet",  preProcess = "scale", 
  trControl = train.control, 
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_seq)
)
sprintf("%.3f", rmse(lasso, test.data))

# Regression Tree

set.seed(100)
tuneGrid = expand.grid(cp = seq(0, 1, by = 0.01))
reg_tree = train(Admission.grade ~ ., data = train.data, 
  method = "rpart",
  trControl = train.control,
  tuneGrid = tuneGrid
)
sprintf("%.3f", rmse(reg_tree, test.data))

# Bagged Regression Tree

set.seed(100)
bagged_reg_tree = train(Admission.grade ~ ., data = train.data, 
  method = "treebag",
  trControl = train.control,
  nbagg = 15,
  control = rpart.control(cp = 0)
)
sprintf("%.3f", rmse(bagged_reg_tree, test.data))

# Simple Random Forest

set.seed(100)
tuneGrid = expand.grid(
  mtry = 1:20,
  splitrule = "variance",
  min.node.size = 5
)
reg_rf = train(Admission.grade ~ ., data = train.data,
  method = "ranger", 
  trControl = train.control,
  importance = "permutation",
  tuneGrid = tuneGrid
)
sprintf("%.3f", rmse(reg_rf, test.data))

# Boosted Random Forest

set.seed(100)
tuneGrid = expand.grid(
  mtry = 1:20,
  splitrule = "variance",
  min.node.size = 5
)
boost_rf = train(Admission.grade ~ ., data = train.data,
  method = "gbm", 
  trControl = train.control,
  tuneLength = 8
)
sprintf("%.3f", rmse(boost_rf, test.data))

# Data Set 2
# Data Wrangling

student2 <- na.omit(student2)
student2 = student2 %>% mutate_if(is.character, as.factor)
set.seed(100)
n = nrow(student2)
train = sample(1:n, n * 0.7)
train.data = student2 %>% slice(train)
test.data = student2 %>% slice(-train)

set.seed(100)
tuneGrid = expand.grid(cp = seq(0, 0.005, by = 0.00001))
dec_tree = train(Target ~ ., data = train.data, 
  method = "rpart",
  trControl = train.control,
  tuneGrid = tuneGrid,
  control = rpart.control(minsplit = 20)
)
predictions <- predict(dec_tree, newdata = test.data)
conf_matrix <- confusionMatrix(predictions, test.data$Target)
sprintf("%.3f", conf_matrix$overall["Accuracy"])

# Bagged Decision Tree

set.seed(100)
bagged_dec_tree = train(Target ~ ., data = train.data, 
  method = "treebag",
  trControl = train.control,
  nbagg = 20,
  control = rpart.control(cp = 0)
)
predictions <- predict(bagged_dec_tree, newdata = test.data)
conf_matrix <- confusionMatrix(predictions, test.data$Target)
sprintf("%.3f", conf_matrix$overall["Accuracy"])

# Simple Random Forest

set.seed(100)
tuneGrid = expand.grid(
  mtry = 1:11,
  splitrule = "gini",
  min.node.size = 5
)
dec_rf = train(Target ~ ., data = train.data,
  method = "ranger", 
  trControl = train.control,
  importance = "permutation",
  tuneGrid = tuneGrid
)
predictions <- predict(dec_rf, newdata = test.data)
conf_matrix <- confusionMatrix(predictions, test.data$Target)
sprintf("%.3f", conf_matrix$overall["Accuracy"])