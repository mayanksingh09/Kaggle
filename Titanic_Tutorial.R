#Titanic Tutorial

## Parts to the script
### 1) Feature Engineering
### 2) Missing Value Imputation
### 3) Prediction

## Loading data

library(ggplot2)
library(ggthemes)
library(scales)
library(dplyr)
library(mice)
library(randomForest)


train <- read.csv('./data/train.csv')
test <- read.csv('./data/test.csv')

full <- bind_rows(train, test)
str(full)


## Feature Engineering

### Title from Names

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title) #Title counts by sex

### Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

### Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

table(full$Sex, full$Title) #Title counts by sex after reassigning


### Finally, grab surname from passenger name
full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

### Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

### Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')


### Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
        geom_bar(stat='count', position='dodge') +
        scale_x_continuous(breaks=c(1:11)) +
        labs(x = 'Family Size') +
        theme_few()

###Noticed that there is a penalty due to family size to singles and to large families

### Discretize family size
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

### Show family size by survival using a mosaic plot
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)


### This variable appears to have a lot of missing values
full$Cabin[1:28]


### The first character is the deck. For example:
strsplit(full$Cabin[2], NULL)[[1]]

### Create a Deck variable. Get passenger deck A - F:
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))


##Missing data handling

### Passengers 62 and 830 are missing Embarkment
full[c(62, 830), 'Embarked']

### Get rid of our missing passenger IDs
embark_fare <- full %>%
        filter(PassengerId != 62 & PassengerId != 830)

### Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
        geom_boxplot() +
        geom_hline(aes(yintercept=80), 
                   colour='red', linetype='dashed', lwd=2) +
        scale_y_continuous(labels=dollar_format()) +
        theme_few()


### Since their fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'


### Show row 1044
full[1044, ]


###Visualize fares among similar passengers
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
        geom_density(fill = '#99d6ff', alpha=0.4) + 
        geom_vline(aes(xintercept=median(Fare, na.rm=T)),
                   colour='red', linetype='dashed', lwd=1) +
        scale_x_continuous(labels=dollar_format()) +
        theme_few()

### Replace missing fare value with median fare for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)



### Show number of missing Age values
sum(is.na(full$Age))

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))


### Set a random seed
set.seed(100)

### Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

### Save the complete output 
mice_output <- complete(mice_mod)

### Plot age distributions to compared original with imputed data
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))


### Replace Age variable from the mice model.
full$Age <- mice_output$Age

### First we'll look at the relationship between age & survival
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
        geom_histogram() + 
        # I include Sex since we know (a priori) it's a significant predictor
        facet_grid(.~Sex) + 
        theme_few()

### Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

### Show counts
table(full$Child, full$Survived)


### Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

### Show counts
table(full$Mother, full$Survived)

### Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

md.pattern(full)


### Split the data back into a train set and a test set
train <- full[1:891,]
test <- full[892:1309,]



### Set a random seed
set.seed(500)

### Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                                 Fare + Embarked + Title + 
                                 FsizeD + Child + Mother,
                         data = train)

###GBM Model

fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)

gbmFit1 <- train(as.factor(Survived) ~ ., data = train[c("Pclass", "Sex", "Age","SibSp", "Parch", "Fare", "Embarked", "Title", "FsizeD", "Child", "Mother", "Survived")], method = "gbm", trControl = fitControl, verbose = FALSE)


### Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)



### Get importance

importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

### Create a rank variable based on importance
rankImportance <- varImportance %>%
        mutate(Rank = paste0('#',dense_rank(desc(Importance))))

### Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
        geom_bar(stat='identity') + 
        geom_text(aes(x = Variables, y = 0.5, label = Rank),
                  hjust=0, vjust=0.55, size = 4, colour = 'red') +
        labs(x = 'Variables') +
        coord_flip() + 
        theme_few()


### Predict using the test set
pred <- predict(rf_model)
table(train$Survived, pred)

predgbm <-predict(gbmFit1, type = "prob")[,2]

pred_test_gbm <- predict(gbmFit1, newdata = test, type = "prob")[,2]

prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

solutiongbm <- data.frame(PassengerID = test$PassengerId, Survived = as.integer(pred_test_gbm >=0.5))

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
write.csv(solutiongbm, file = 'gbmfe_Solution.csv', row.names = F)


## XGBoost model

### Recoding data to make it numeric

train[c("Pclass", "Sex", "Age","SibSp", "Parch", "Fare", "Embarked", "Title", "FsizeD", "Child", "Mother", "Survived")]

head(train$Title)


full2 <- full


full2$Sex <- ifelse(full2$Sex == 'male',1,0)
#y <- recode(labels$labels,"'True'=1; 'False'=0)

full2$Embarked <- ifelse(full2$Embarked == "C",1, ifelse(full2$Embarked == "Q", 2, 3))

full2$Title <- ifelse(full2$Title == "Mr", 1, ifelse(full2$Title == "Mrs", 2, ifelse(full2$Title == "Miss", 3, ifelse(full2$Title == "Master", 4, 5))))

full2$FsizeD <- ifelse(full2$FsizeD == "singleton", 1, ifelse(full2$FsizeD == "small", 2, 3))
full2$Child <- ifelse(full2$Child == "Child", 1, 0)
full2$Mother <- ifelse(full2$Mother == "Mother", 1, 0)

train2 <- full2[1:891,]
test2 <- full2[892:1309,]

train_reqd <- train2[c("Pclass", "Sex", "Age","SibSp", "Parch", "Fare", "Embarked", "Title", "FsizeD", "Child", "Mother")]
str(train_reqd)
y <- train$Survived
test_reqd <- test2[c("Pclass", "Sex", "Age","SibSp", "Parch", "Fare", "Embarked", "Title", "FsizeD", "Child", "Mother")]

xgb <- xgboost(data = data.matrix(train_reqd), label = y, eta = 0.1, max_depth = 15, nround=25, subsample = 0.5, nthread = 3, objective = "binary:logistic")


pred <- predict(xgb, data.matrix(train_reqd))
length(pred)

pred_test <- predict(xgb, data.matrix(test_reqd))

outputxgb <- data.frame(PassengerID = test$PassengerId, Survived = as.integer(pred_test >= 0.5))


write.csv(outputxgb, file = "xgb_fe.csv", row.names = F)
