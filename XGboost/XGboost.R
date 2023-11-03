library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(rpart.plot)
library(tidyverse)
set.seed(100)
setwd("D:\\covid\\R")
Data <- read_csv("Data_2.csv")
summary(Data) 

# Determine row to split on: split
parts = createDataPartition(Data$Mortality, p = .8, list = F)
train = Data[parts, ]
test = Data[-parts, ]
#define predictor and response variables in training set
train_x = data.matrix(train[, -1])
train_y = data.matrix(train[,1])

#define predictor and response variables in testing set
test_x = data.matrix(test[, -1])
test_y = data.matrix(test[, 1])

#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)

#define final model
model_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)

summary(model_xgboost)

#use model to make predictions on test data
pred_y = predict(model_xgboost, xgb_test)

# performance metrics on the test data

mean((test_y - pred_y)^2) #mse - Mean Squared Error

caret::RMSE(test_y, pred_y) #rmse - Root Mean Squared Error

y_test_mean = mean(test_y)

# Calculate total sum of squares
tss =  sum((test_y - y_test_mean)^2 )
# Calculate residual sum of squares
rss =  sum(resid(model_xgboost)^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')

x = 1:length(test_y)                   # visualize the model, actual and predicted data
plot(x, test_y, col = "red", type = "l")
lines(x, pred_y, col = "blue", type = "l")
legend(x = 1, y = 38,  legend = c("original test_y", "predicted test_y"), 
       col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))

err <- round(mean(as.numeric(pred_y > 0.5) != test_y), digits=4)
print(paste("accuracy=", (1-err)*100, "%"))

# Compute feature importance matrix
importance_matrix = xgb.importance(colnames(xgb_train), model = model_xgboost)
importance_matrix

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])
