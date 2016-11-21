### Decision tree with rpart on the imbalanced data set


## Clear memmory
rm(list=ls(all=TRUE))

## loading required packages
library(readxl)
library(ggplot2)
library(ROCR)
library(rpart)
library(randomForest)

## loading required functions
source('metrics.R')


### Read the data from Excel file
defaultdata <- read_excel("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients.xlsx")
#View(defaultdata)
#df = read.xls("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients_old.xls", sheet = 1, header = TRUE)
defaultdata = data.frame(defaultdata[,-1])



## variable names in the data set
varNames  = names(defaultdata)

## column number of the response variable default_class
class_colnum = which(varNames=='default_class')


##  ndfper is the percentage of the people did not defaulted in the data set
ndfper = length(which(defaultdata$default_class==0))/nrow(defaultdata)*100
print(ndfper) ## 77.88% of people did not defaulted. This indicates that the data set is not a balanced data set



### Note that all the variables in defaultdata are numeric
## convert the catogorical variales to factors
defaultdata$default_class = factor(defaultdata$default_class, levels = 0:1, labels = c('No','yes'))
defaultdata$SEX = factor(defaultdata$SEX)
defaultdata$EDUCATION = factor(defaultdata$EDUCATION)
defaultdata$MARRIAGE = factor(defaultdata$MARRIAGE)

###  to check if the ordinal variabls considered numeric make a different in the model
### this does not make any difference
# defaultdata$PAY_0 = factor(defaultdata$PAY_0)
# efaultdata$PAY_2 = factor(defaultdata$PAY_2)
# defaultdata$PAY_3 = factor(defaultdata$PAY_3)
# defaultdata$PAY_4 = factor(defaultdata$PAY_4)
# defaultdata$PAY_5 = factor(defaultdata$PAY_5)
# defaultdata$PAY_6 = factor(defaultdata$PAY_6)


# validation set approach
# create a training set that randomly samples from 2/3 of the available data 
set.seed(10)
numdata = 2*nrow(defaultdata)/3
train = sample(1:nrow(defaultdata), numdata)

# the test rows are those not in train
test = (-train)

# test data
y.test = defaultdata$default_class[test]
data.test = defaultdata[test, ]

## training data
y.train = defaultdata$default_class[train]
data.train = defaultdata[train, ]



### fit the tree on training data
mycontrol = rpart.control(cp = 0.00, xval = 10)
tree.fit = rpart(default_class~., data = data.train, method = 'class', control = mycontrol)
pruned.fit<- prune(tree.fit, cp = tree.fit$cptable[which.min(tree.fit$cptable[,"xerror"]),"CP"])
print(pruned.fit)
#plot(pruned.fit)
#text(pruned.fit)

## confusion matrix
tree.pred = predict(tree.fit, newdata = data.test,type = 'cl')
tabl = table(tree.pred, y.test)
mean(tree.pred==y.test)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)
    
print(tree.fit$cptable)
#plotcp(tree.fit)
x = as.numeric(tree.fit$cptable[,2])
y = as.numeric(tree.fit$cptable[,4])
plot(x, y, col='red', type ='l', lwd=2, xlab = 'number of spits in the tree', ylab = 'error')


##plot the ROC curve
tree.pred = predict(tree.fit, newdata = data.test,type = 'prob')[,2]
pred = prediction(tree.pred, y.test)
tree.perf = performance(pred,"tpr","fpr")
plot(tree.perf,col=3,lwd=1,xlab="False positive rate",ylab="True positive rate", main="ROC Curve")
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)




###################################################### Baggibg model
numpred = ncol(data.train)-1
forest.fit <- randomForest(default_class ~ .,   data = defaultdata, subset = train, mtry = sqrt(numpred), ntree = 500, importance = TRUE)
print(forest.fit) # view results 
importance(forest.fit) # importance of each predictor

forest.pred = predict(forest.fit, newdata = eval.data,type = 'cl')
tabl = table(forest.pred, y.eval.data)
mean(forest.pred==y.eval.data)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)

##plot the ROC curve
forest.pred = predict(forest.fit, newdata = eval.data,type = 'prob')[,2]
pred = prediction(forest.pred, y.eval.data)
bag.perf = performance(pred,"tpr","fpr")
plot(bag.perf,col= 2,lwd=1)
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)

### OOB error for numtree = 10, 50, 100, 500,600, 700, 800, 900, 1000
##  23.76%, 19.14%, 18.69%, 18.66%, 18.62%, 18.47%, 18.57%, 18.54%, 18.59%

numtree = c(10, 50, 100, 500,600, 700, 800, 900, 1000)
ooberror = c(23.76, 19.14, 18.69, 18.66, 18.62, 18.47, 18.57, 18.54, 18.59)
plot(numtree, ooberror, col='red', type='b',pch = 19, xlab='number of trees', ylab='OOB error percentage')


######################## Random forest model
numpred = ncol(data.train)-1
forest.fit <- randomForest(default_class ~ .,   data = defaultdata, subset = train, mtry = sqrt(numpred), ntree = 10, importance = TRUE)

print(forest.fit) # view results 
importance(forest.fit) # importance of each predictor

forest.pred = predict(forest.fit, newdata = data.test,type = 'cl')
tabl = table(forest.pred, y.test)
mean(forest.pred==y.test)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)


##plot the ROC curve
forest.pred = predict(forest.fit, newdata = data.test,type = 'prob')[,2]
pred = prediction(forest.pred, y.test)
forest.perf = performance(pred,"tpr","fpr")
plot(forest.perf,col=4 )
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)
#varImpPlot(forest.fit)

### plot the tuning parameter val with num trees
numtree = c(10, 50, 100, 500, 600, 700, 800, 900, 1000)
### OOB error for different p values  where p is num predictors
### note that the case mtry = p is Bagging
## mtry is the num of variabes at each spit
e1 = c(23.82, 19, 18.38,  18.33, 18.41,18.38,18.38, 18.43, 18.43 )##sqrt(p)
e2 = c(23.58, 19.12, 18.85, 18.43, 18.46, 18.57, 18.46, 18.43, 18.49 )##p/2
e3 = c(23.42, 19.02, 18.75, 18.55, 18.44, 18.48,  18.56, 18.5, 18.75)##p/3
e4 = c(23.76, 19.14, 18.69, 18.66, 18.62, 18.47, 18.57, 18.54, 18.59)##p
plot(numtree, e1, col =2, type='b',pch =19, xlab='number of trees', ylab = 'OOB error percentage')
lines(numtree, e2, col =3, type='b',pch =18)
lines(numtree, e3, col =4, type='b',pch =17)
lines(numtree, e4, col =6, type='b',pch =16)
legend(x="topright",legend=c("sqrt(p)","p/2","p/3","p"),col=c(2,3,4,6),lty=c(1,1,1,1),pch =c(19,18,17,16),cex=1.0)








##################################################### Boosting model
library(gbm)
boost.fit = gbm((unclass(default_class)-1)~., data = data.train, distribution = 'bernoulli', n.tree = 5000, interaction.depth = 4)
# shrinkage = 0.001
#summary(boost.fit)


boost.pred = predict(boost.fit, newdata = data.test, n.tree = 5000, type = 'response')
boost.predclass = rep("0", length(y.test))
boost.predclass[boost.pred > 0.5]="1"
tabl = table(boost.predclass, (unclass(y.test)-1))
mean(boost.predclass==(unclass(y.test)-1))
tp = tabl[1,1]
fp = tabl[1,2]
fn = tabl[2,1]
tn = tabl[2,2]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)



pred = prediction(boost.pred, y.test)
boost.perf = performance(pred,"tpr","fpr")
plot(boost.perf,col=6,lwd=1)
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)


