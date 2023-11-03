library('randomForest')
library(xgboost)
#read data
setwd("D:\\covid\\R")
Data <- read.csv("Data_2.csv")
head(data)


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

rf <- randomForest(x=train_x,train_y,  mtry=4, ntree=2001, importance=TRUE)

rf
plot(rf)

result <- data.frame(test_y, predict(rf, test_x, type = "response"))
result
plot(result)

pred_y = predict(rf, test)

mean((test_y - pred_y)^2) #mse - Mean Squared Error

caret::RMSE(test_y, pred_y) #rmse - Root Mean Squared Error

y_test_mean = mean(test_y)

# Calculate total sum of squares
tss =  sum((test_y - y_test_mean)^2 )
# Calculate residual sum of squares
rss =  sum(resid(rf)^2)
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



