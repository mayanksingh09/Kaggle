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
