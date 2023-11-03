library(tidyverse)
library(sjPlot)
library(lme4) 
library(caret)
library(Metrics)
library(MASS)

set.seed(123)
Data <- read.csv("Data_2.csv")
summary(Data)  
head(Data)
str(Data)
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

dim(train)
dim(test)

model <- glm(Mortality~+Age+Male+Platelet+Creatinine+White.Blood.Cell+Diabetes+Cardiovascular+Lymphocyte+Fever+Neutrophil, family = gaussian, data = train)
summary(model)

pred <- predict(model, test, type = 'response')

x = 1:length(test_y)                   # visualize the model, actual and predicted data
plot(x, test_y, col = "red", type = "l")
lines(x, pred, col = "blue", type = "l")
legend(x = 1, y = 38,  legend = c("original test_y", "predicted test_y"), 
       col = c("red", "blue"), box.lty = 1, cex = 0.8, lty = c(1, 1))



# performance metrics on the test data

mean((test_y - pred)^2) #mse - Mean Squared Error

caret::RMSE(test_y, pred) #rmse - Root Mean Squared Error

y_test_mean = mean(test_y)

# Calculate total sum of squares
tss =  sum((test_y - y_test_mean)^2 )
# Calculate residual sum of squares
rss =  sum(resid(model)^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')


err <- round(mean(as.numeric(pred > 0.5) != test_y), digits=4)
print(paste("accuracy=", (1-err)*100, "%"))

