library(caTools)
library(data.table)
library(car)
library(pROC)
library(caret)
library(ResourceSelection)
library(mice)
library(randomForest)

#Titanic Logistic Regression Model
setwd("C:/Users/Mayank/Documents/kaggle/Titanic")

train <- read.csv("./data/train.csv")

md.pattern(train) #Table showing missing values
newdata <- mice(data = train[c("Pclass", "Sex", "Age", "Parch", "Fare")], m = 5, maxit = 50, method = "pmm") #can use pmm or fastpmm also

trainimputed <- complete(newdata, 1)

train$Age <- trainimputed$Age


#Multiple imputation on Test data
test <- read.csv("./data/test.csv")

newdatatest <- mice(data = test[c("Pclass", "Sex", "Age", "Parch", "Fare")], m = 5, maxit = 50, method = "pmm") #can use pmm or fastpmm also

testimputed <- complete(newdatatest, 1)

test$Age <- testimputed$Age
test$Fare <- testimputed$Fare

titanicmod1 <- glm(Survived ~  Pclass + Sex + Age + SibSp + Parch + Fare  + Embarked, data = train, family = binomial)

#Stepwise elimination
titanicmod2 <- step(titanicmod1)

summary(titanicmod2)


#Final model
titanicmod3 <- glm(Survived ~ as.factor(Pclass) + Sex  +Age + SibSp, data = train, family = binomial)


#Prediction on training set
prediction <- predict(titanicmod2, type = "response")

table(train$Survived, prediction > 0.5)
(507 + 221)/nrow(train)

predicttest <- predict(titanicmod2, type = "response", newdata = test)

test$Survived <- as.integer(predicttest >= 0.6)

submission <- test[c("PassengerId", "Survived")]

#write.csv(submission, file = "submission.csv")

##RANDOM FOREST Model
test$Pclass <- as.factor(test$Pclass)
train$Pclass <- as.factor(train$Pclass)

titanicRF <- randomForest(Survived ~., data = train[c("Pclass", "Sex", "Age","SibSp", "Survived", "Parch", "Fare", "Embarked")])

predictRFtrain <- predict(titanicRF)
table(train$Survived, predictRFtrain >= 0.6)

(500 + 240)/nrow(train)

str(test)
str(train)

predictRF <- predict(titanicRF, newdata = test)

test$Survived <- predictRF
outputRF <- test[c("PassengerId", "Survived")]

#write.csv(outputRF, file = "OutputRF.csv")


##GBM Model

fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)

gbmFit1 <- train(as.factor(Survived) ~., data= train[c("Pclass", "Sex", "Age","SibSp", "Survived", "Parch", "Fare", "Embarked")], method = "gbm", trControl = fitControl, verbose = F)

gbmpred <- predict(gbmFit1, type = "prob")

table(train$Survived, gbmpred[,2] >= 0.5)
(505 + 258)/nrow(train)

test$Survived <- as.integer(predict(gbmFit1, type = "prob", newdata = test)[,2] >= 0.6)

test$PassengerId
gbmsub <- test[c("PassengerId", "Survived")]

write.csv(gbmsub, file = "gbmsub.csv")


#XGBoost Algorithm
library(Matrix)


sparse_matrix <- sparse.model.matrix(Survived ~ .-1, data = train[c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked")])

##colnames(train)

##titanicmod1 <- glm(Survived ~  Pclass + Sex + Age + SibSp + Parch + Fare  + Embarked, data = train, family = binomial)

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

str(train)
train$SexNew <- ifelse(train$Sex == "male", 1, 0)
test$SexNew <- ifelse(test$Sex == "male", 1, 0)


train_reqd <- train[c("Pclass", "SexNew", "Age", "SibSp")]
test_reqd <- test[c("Pclass", "SexNew", "Age", "SibSp")]

y <- train$Survived

xgb <- xgboost(data = data.matrix(train_reqd), max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")


xgb2 <- xgboost(data = data.matrix(train_reqd), 
               label = y, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               num_class = 12,
               nthread = 3
)



summary(xgb2)

pred <- predict(xgb, data.matrix(train_reqd))
pred2 <- predict(xgb, data.matrix(train_reqd))

pred_test <- predict(xgb, data.matrix(test_reqd))
pred_test2 <- predict(xgb2, data.matrix(test_reqd))

output <- cbind(test$PassengerId,as.integer(pred_test2 >= 0.5))

colnames(output) <- c("PassengerId", "Survived")

write.csv(output, file = "./data/titanicxgb.csv", row.names = FALSE)

