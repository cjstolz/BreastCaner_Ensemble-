# BreastCaner_Ensemble-
Testing of various classification models and some work using ensembles
---
title: "Project Two"
author: "Cody Stolz"
date: "3/2/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#Load these necesary packages, use install.packages if you don't alreadt have them installed

```{r load packages}
library(mlbench)
library(e1071)
library(klaR)
library(nnet)
library(rpart)
library(party)
library(ipred)
library(ROCR)
library(randomForest)
library(caret)
library(caretEnsemble)
library(pROC)
library(plotROC)
```
#we will use mlbench built-in dataset BreastCancer

```{r load data}
data("BreastCancer")
```

# Breastcancer dataset model performance and esemble code

#Remove missing values

```{r omit missing values}
BreastCancer <- na.omit(BreastCancer) 
BreastCancer$Id <- NULL 
```


#Then we need partition the data in order to create training and validation data sets

```{r partition data}
train.index <- sample( c( 1: dim(BreastCancer)[ 1]), dim(BreastCancer)[ 1]* 0.6) #separate, 60% of data to training, 40% to validation 
train.df <- BreastCancer[ train.index, ] 
valid.df <- BreastCancer[-train.index, ]
```

#Lets take a took at various models we can use

#SVM is a classification and regression model that can preform both linear and non-linear classifications  

```{r svm model}
mysvm <- svm(Class ~ ., BreastCancer)
mysvm.pred <- predict(mysvm, BreastCancer)
table(mysvm.pred, BreastCancer$Class) #Code to look at a table of the predictions
```
#now lets look at the models preformance 

```{r svm preformance}
svm1 <- svm(Class ~ ., data = train.df) 
svmpred <- predict(svm1, valid.df, type = "class") 
confusionMatrix(svmpred, valid.df$Class) #confusion matrix allows to assess accuracy of the model predictions against the validation data
```
#We can see the accuracy of the model is ~96%

#Next we can see a NaiveBayes model which applies Bayes' theorem to features 

```{r nb model}
mynb <- NaiveBayes(Class ~ ., BreastCancer)
mynb.pred <- predict(mynb,BreastCancer)
table(mynb.pred$class, BreastCancer$Class)
```

#Naive Bayes preformance

```{r }
nb1 <- naiveBayes(Class ~ ., data = train.df) 
nbpred <- predict(nb1, valid.df, type = "class") 
confusionMatrix(nbpred, valid.df$Class)
```

#Neural netword useful for predicting categorical values 

```{r nnet}
mynnet <- nnet(Class ~ ., BreastCancer, size=1)
mynnet.pred <- predict(mynnet,BreastCancer,type="class")
table(mynnet.pred,BreastCancer$Class)
```

#nnet preformance

```{r nnet preformance}
nn <- nnet(Class ~ ., train.df, size=1) 
nnpred <- predict(nn, valid.df, type = "class") 
confusionMatrix(as.factor(nnpred), valid.df$Class)
```

#Regression tree through rpart

```{r regression tree}
mytree <- rpart(Class ~ ., BreastCancer)
plot(mytree); text(mytree) 
summary(mytree)
mytree.pred <- predict(mytree,BreastCancer,type="class")
table(mytree.pred,BreastCancer$Class)
```
#Code to look at the rpart preformance 

```{r look at the preformance of rpart}
tr <- rpart(Class ~ ., data = train.df) 
pred <- predict( tr, valid.df, type = "class") 
confusionMatrix(pred, valid.df$Class)
```
#Here are a few more models that you can use 

#leave one out cross validation for accuracy or predictive model 

```{r loocv cross validation}
ans <- numeric(length(BreastCancer[,1]))
for (i in 1:length(BreastCancer[,1])) {
  mytree <- rpart(Class ~ ., BreastCancer[-i,])
  mytree.pred <- predict(mytree,BreastCancer[i,],type="class")
  ans[i] <- mytree.pred
}
ans <- factor(ans,labels=levels(BreastCancer$Class))
table(ans,BreastCancer$Class)
```

```{r loocv cross validation with confusion matrix}
ans1 <- numeric(length(train.df[,1]))
for (i in 1:length(train.df[,1]))
  ans1 <- numeric(length(valid.df[,1]))
for (i in 1:length(valid.df[,1])){
  mytree1 <- rpart(Class ~ ., train.df[-i,])
  mytree.pred1 <- predict(mytree1,valid.df[i,],type="class")
  ans1[i] <- mytree.pred1
}
ans1 <- factor(ans1,labels=levels(train.df$Class))
confusionMatrix(ans1, valid.df$Class)
```


#Regularised Discriminant Analysis code

```{r regularized}
myrda <- rda(Class ~ ., BreastCancer)
myrda.pred <- predict(myrda, BreastCancer)
table(myrda.pred$class,BreastCancer$Class) #switch to numeric
```

```{r rda with confusion matrix}
myrda1 <- rda(Class ~ ., train.df)
rda.pred <- predict(myrda1, valid.df, type="class")
confusionMatrix(rda.pred$class, valid.df$Class)
```


#run a random forrest code

```{r random forrest} 
myrf <- randomForest(Class ~ .,BreastCancer)
myrf.pred <- predict(myrf, BreastCancer)
table(myrf.pred, BreastCancer$Class)
```

```{r random forrest with confusion matrix} 
rf <- randomForest(Class ~ .,train.df)
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$Class)
```

#Lets look at the results of the various models
#SVM .959
#NB .978
#NNET .96
#Rpart .956
#Loocv .952
#RDA .98
#RF .974



#Now that we've built some models and evaluated their performance, lets look at conbiing models in an ensemble fashion. caretEnsemble makes this very easy as you'll see below. 

#We have to use built-in caret models for this type of ensemble. If you don't know which models you can use, reference the caret package help information.

#For our ensemble, let's use rpart, glm, knn, and svmRadial 

```{r predictions}
set.seed(100)

control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE) #we're use the repeated cross validation method here

algorithms_to_use <- c('rpart', 'glm', 'knn', 'svmRadial')

stacked_models <- caretList(Class ~., data=BreastCancer, trControl=control_stacking, methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

summary(stacking_results)
```


#We can see each models performance across all iterations 

#We can easily plot them using xyplot

```{r plot models}
xyplot(resamples(stacked_models))
```


#now lets combine all four models in an ensemble for a better prediction
```{r combine models}
stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions=TRUE, classProbs=TRUE)

set.seed(100)

glm_stack <- caretStack(stacked_models, method="glm", metric="Accuracy", trControl=stackControl)

print(glm_stack)
```
#look again at the accuracy 
```{r plot models 2}
xyplot(resamples(stacked_models))
```

#for the majority rule approach, we can see that on average knn and rpart have the highest accuracy from the above models. Lets develop an ensemble method that combines these two methods

```{r predictions 2}
set.seed(100)

control_stacking <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE) #we're use the repeated cross validation method here

algorithms_to_use <- c('knn', 'svmRadial')

stacked_models <- caretList(Class ~., data=BreastCancer, trControl=my_control, methodList=algorithms_to_use)

stacking_results <- resamples(stacked_models)

summary(stacking_results)
```

#Now lets combine using the two models
```{r combine models again}
stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions=TRUE, classProbs=TRUE)

set.seed(100)

glm_stack <- caretStack(stacked_models, method="glm", metric="Accuracy", trControl=stackControl)

print(glm_stack)
```

#It looks like our ensemble is slightly better using the best two models only





#Below are some more methods to combine classifers including bootstrap and bagging 
```{r sample and replace}
ind <- sample(2, nrow(BreastCancer), replace = TRUE, prob=c(0.8, 0.2))
```

```{r predict classes}
x.rp <- rpart(Class ~ ., data=BreastCancer[ind == 1,])
# predict classes for the evaluation data set
x.rp.pred <- predict(x.rp, type="class", newdata=BreastCancer[ind == 2,])
# score the evaluation data set (extract the probabilities)
x.rp.prob <- predict(x.rp, type="prob", newdata=BreastCancer[ind == 2,])
```

```{r inference model}
x.ct <- ctree(Class ~ ., data=BreastCancer[ind == 1,])
x.ct.pred <- predict(x.ct, newdata=BreastCancer[ind == 2,])
x.ct.prob <-  1- unlist(treeresponse(x.ct, BreastCancer[ind == 2,]), use.names=F)[seq(1,nrow(BreastCancer[ind == 2,])*2,2)]
```

```{r create model using random forest}
x.cf <- cforest(Class ~ ., data=BreastCancer[ind == 1,], control = cforest_unbiased(mtry = ncol(BreastCancer)-2))
x.cf.pred <- predict(x.cf, newdata=BreastCancer[ind == 2,])
x.cf.prob <-  1- unlist(treeresponse(x.cf, BreastCancer[ind == 2,]), use.names=F)[seq(1,nrow(BreastCancer[ind == 2,])*2,2)]
```

```{r bootstrap model}
x.ip <- bagging(Class ~ ., data=BreastCancer[ind == 1,])
x.ip.prob <- predict(x.ip, type="prob", newdata=BreastCancer[ind == 2,])
```

```{r SVM}
x.svm.tune <- tune(svm, Class~., data = BreastCancer[ind == 1,],
                   ranges = list(gamma = 2^(-8:1), cost = 2^(0:4)),
                   tunecontrol = tune.control(sampling = "fix"))
```


```{r display SVM}
x.svm.tune
```

```{r manually enter perameters}
x.svm <- svm(Class~., data = BreastCancer[ind == 1,], cost=4, gamma=0.0625, probability = TRUE)
x.svm.prob <- predict(x.svm, type="prob", newdata=BreastCancer[ind == 2,], probability = TRUE)
```

```{r png output}
png(filename="roc_curve_5_models.png", width=700, height=700)
```

# create an ROCR prediction object from rpart() probabilities
```{r create a ROCR prediction}
x.rp.prob.rocr <- prediction(x.rp.prob[,2], BreastCancer[ind == 2,'Class'])
# prepare an ROCR performance object for ROC curve (tpr=true positive rate, fpr=false positive rate)
x.rp.perf <- performance(x.rp.prob.rocr, "tpr","fpr")
# plot it
plot(x.rp.perf, col=2, main="ROC curves comparing classification performance of five machine learning models")
legend(0.6, 0.6, c('rpart', 'ctree', 'cforest','bagging','svm'), 2:6)
```

```{r c tree}
x.ct.prob.rocr <- prediction(x.ct.prob, BreastCancer[ind == 2,'Class'])
x.ct.perf <- performance(x.ct.prob.rocr, "tpr","fpr")
```


```{r add colors to exsisting chart}
# add=TRUE draws on the existing chart 
plot(x.rp.perf, col=2, main="ROC curves comparing classification performance of five machine learning models")
legend(0.6, 0.6, c('rpart', 'ctree', 'cforest','bagging','svm'), 2:6)
plot(x.ct.perf, col=3, add=TRUE)
```

```{r c forrest -}
x.cf.prob.rocr <- prediction(x.cf.prob, BreastCancer[ind == 2,'Class'])
x.cf.perf <- performance(x.cf.prob.rocr, "tpr","fpr")
```

```{r plot with c forrest}
# add=TRUE draws on the existing chart 
plot(x.rp.perf, col=2, main="ROC curves comparing classification performance of five machine learning models")
legend(0.6, 0.6, c('rpart', 'ctree', 'cforest','bagging','svm'), 2:6)
plot(x.ct.perf, col=3, add=TRUE)
plot(x.cf.perf, col=4, add=TRUE)
```

```{r with bagging}
x.ip.prob.rocr <- prediction(x.ip.prob[,2], BreastCancer[ind == 2,'Class'])
x.ip.perf <- performance(x.ip.prob.rocr, "tpr","fpr")
```

```{r plot with bagging}
# add=TRUE draws on the existing chart 
plot(x.rp.perf, col=2, main="ROC curves comparing classification performance of five machine learning models")
legend(0.6, 0.6, c('rpart', 'ctree', 'cforest','bagging','svm'), 2:6)
plot(x.ct.perf, col=3, add=TRUE)
plot(x.cf.perf, col=4, add=TRUE)
plot(x.ip.perf, col=5, add=TRUE)
```

```{r SVM}
x.svm.prob.rocr <- prediction(attr(x.svm.prob, "probabilities")[,2], BreastCancer[ind == 2,'Class'])
x.svm.perf <- performance(x.svm.prob.rocr, "tpr","fpr")
```

```{r plot with bagging}
# add=TRUE draws on the existing chart 
plot(x.rp.perf, col=2, main="ROC curves comparing classification performance of five machine learning models")
legend(0.6, 0.6, c('rpart', 'ctree', 'cforest','bagging','svm'), 2:6)
plot(x.ct.perf, col=3, add=TRUE)
plot(x.cf.perf, col=4, add=TRUE)
plot(x.ip.perf, col=5, add=TRUE)
plot(x.svm.perf, col=6, add=TRUE)
```

























































