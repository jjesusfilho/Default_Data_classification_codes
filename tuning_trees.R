### Decision tree with rpart on the imbalanced data set
### with tuning for ROC curve and sensitivity using 10-fold cross validation


## Clear memmory
rm(list=ls(all=TRUE))

## loading required packages
library(readxl)
library(ggplot2)
library(ROCR)
library(rpart)
library(randomForest)
library(gbm)

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



# set aside a  small evaluation set from the test data to be used for
#developing post-processing techniques, such as alternative probabilitycutoffs (numdata = 2000)
set.seed(1)
numdataeval = nrow(data.test)/5
eval = sample(1:nrow(data.test), numdataeval)
eval.data = data.test[eval, ]
y.eval.data = data.test$default_class[eval]



######################## Decision tree
## Devide the training set to 10 equal parts for k-fold cross validation
numfolds = 10 
foldndata = numdata/numfolds

rowsmat = matrix(0, nrow = foldndata, ncol = numfolds)
totaldata = numdata
newrows = c(1:numdata)
for ( i in 1:numfolds)
{
  vec = sample(1:totaldata, foldndata, replace=FALSE)
  rowsmat[, i] = vec
  newrows = newrows[-vec]
  totaldata = length(newrows)
}

### We tune the tree with 10-fold cross validation
## k-fold cross validation on training set to pick the best cp value for pruning the tree
cpval = c(1.790095e-01,  2.498864e-03, 2.385279e-03, 2.044525e-03, 1.135847e-03,8.329547e-04, 5.841501e-04, 4.922005e-04, 1.817356e-04)
numsplits = c(0, 1,5, 7, 10, 30,100, 149, 412)
n = length(cpval)
cverror = rep(0,n)##store the sensitivity accuracy
AUC = rep(0,n)
acc = rep(0,n)# accuracy rate
for(r in c(1:n))
{
  kfoldcv.errorvec = rep(0,numfolds)
  AUCvec = rep(0,numfolds)
  accvec = rep(0,numfolds)
  for ( j in 1:numfolds)
  {
    indexvec = as.vector(rowsmat[, j])
    x.traincv = data.train[-indexvec, ]
    y.traincv = y.train[-indexvec]
    x.testcv = data.train[indexvec, ]
    y.testcv = y.train[indexvec]
    
    mycontrol = rpart.control(cp = cpval[r], xval = 10)
    tree.fit = rpart(default_class~., data = x.traincv, method = 'class', control = mycontrol)
    tree.pred = predict(tree.fit, newdata = x.testcv, type = 'cl')
    tabl = table(tree.pred, y.testcv)
    mean(tree.pred==y.testcv)
    tp = tabl[2,2]
    fp = tabl[2,1]
    fn = tabl[1,2]
    tn = tabl[1,1]
    metricvec = metricfun (tp, fp, fn, tn)
    tree.pred = predict(tree.fit, newdata = x.testcv, type = 'prob')[,2]
    pred = prediction(tree.pred, y.testcv)
   
    perf_auc =  performance(pred, measure = "auc")
    AUCvec[j] = as.numeric(perf_auc@y.values)
    kfoldcv.errorvec[j] = metricvec[2]
    accvec[j] = metricvec[1]
  } 
  cverror[r] = mean(kfoldcv.errorvec)
  AUC[r] = mean(AUCvec)
  acc[r] = mean(accvec)
}


print(cverror)
print(AUC)
print(acc)
plot(numsplits, AUC, type='l',col = 3, lwd = 2, xlab='number of splits in the tree', ylab='AUC of ROC')
optcp = cpval[which.max(AUC)]


### fit the tree on the traing data
mycontrol = rpart.control(cp = optcp, xval = 10)
tree.fit = rpart(default_class~., data = data.train, method = 'class', control = mycontrol)

### test the Decision tree model on the small evaluation set
tree.pred = predict(tree.fit, newdata = eval.data,type = 'cl')
tabl = table(tree.pred, y.eval.data)## confusion matrix
mean(tree.pred==y.eval.data)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)

##plot the ROC curve
tree.pred = predict(tree.fit, newdata = eval.data,type = 'prob')[,2]
pred = prediction(tree.pred, y.eval.data)
tree.perf = performance(pred,"tpr","fpr")
plot(tree.perf,col=2,lwd=1,xlab="False positive rate",ylab="True positive rate", main="ROC Curve")
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)


############################################## Bagging and random forest

## Devide the training set to 2 equal parts for k-fold cross validation
numfolds = 2
foldndata = numdata/numfolds

rowsmat = matrix(0, nrow = foldndata, ncol = numfolds)
totaldata = numdata
newrows = c(1:numdata)
for ( i in 1:numfolds)
{
  vec = sample(1:totaldata, foldndata, replace=FALSE)
  rowsmat[, i] = vec
  newrows = newrows[-vec]
  totaldata = length(newrows)
}

### We tune the bagged trees with 10-fold cross validation
## k-fold cross validation on training set to pick the best number of trees for bagging
ntreeval = c(1000)
n = length(ntreeval)
cverror = rep(0,n)##store the sensitivity accuracy
AUC = rep(0,n)
for(r in c(1:n))
{
  kfoldcv.errorvec = rep(0,numfolds)
  AUCvec = rep(0,numfolds)
  for ( j in 1:numfolds)
  {
    indexvec = as.vector(rowsmat[, j])
    x.traincv = data.train[-indexvec, ]
    y.traincv = y.train[-indexvec]
    x.testcv = data.train[indexvec, ]
    y.testcv = y.train[indexvec]
    numpred = ncol(data.train)-1
    forest.fit <- randomForest(default_class ~ .,   data = x.traincv, mtry = numpred/3, ntree = ntreeval[r], importance = TRUE)
    forest.pred = predict(forest.fit, newdata = x.testcv,type = 'cl')
    tabl = table(forest.pred, y.testcv)
    mean(forest.pred==y.testcv)
    tp = tabl[2,2]
    fp = tabl[2,1]
    fn = tabl[1,2]
    tn = tabl[1,1]
    metricvec = metricfun (tp, fp, fn, tn)
    forest.pred = predict(forest.fit, newdata = x.testcv, type = 'prob')[,2]
    pred = prediction(forest.pred, y.testcv)
    
    perf_auc =  performance(pred, measure = "auc")
    AUCvec[j] = as.numeric(perf_auc@y.values)
    kfoldcv.errorvec[j] = metricvec[2]
  } 
  cverror[r] = mean(kfoldcv.errorvec)
  AUC[r] = mean(AUCvec)
}
print(cverror)
print(AUC)

### Bagging results
## cross val AUC for ntree = 10,50,100,500, 600, 700, 800, 900, 1000 respectively
## 0.7295364, 0.7489307,0.7541596, 0.7549247, 0.7557496,0.7554537, 0.7560308,  0.7561025,  0.7559163 


## cross val sensitivities for ntree = 10,50, 100,500,600, 700, 800, 900, 1000 respectively
## 0.9119249, 0.932913,0.9364334,0.9376483, 0.9385444,0.9381615, 0.937521, 0.9377768,  0.9386099

numtree = c(10,50,100,500, 600, 700, 800, 900, 1000)
auc = c(0.7295364, 0.7489307,0.7541596, 0.7549247, 0.7557496,0.7554537, 0.7560308,  0.7561025,  0.7559163)
plot(numtree, auc, col ='red',type='b',pch = 19,xlab ='number of trees', ylab='AUC of ROC')


### Random forest results
e11 = c(0.7235393, 0.7545002, 0.7544766, 0.7618233, 0.7608201,0.7612939,0.7614622,  0.7617306,  0.7611353) # mtry = sqrt(p)
e12 = c(0.7218792,0.7510094,0.7516026, 0.7564247,0.7575124,0.7578132, 0.7578987,0.7578224 ,0.7573048) # mtry = p/2
e13 = c(0.7243849,0.7514034, 0.7556856,0.7584692,  0.7596073,0.7592237,  0.7590585, 0.7588884, 0.7594447 )#mtry = p/3
e14 = c(0.7295364, 0.7489307,0.7541596, 0.7549247, 0.7557496,0.7554537, 0.7560308,  0.7561025,  0.7559163)# mtry = p

### sensitivities
#0.9205639, 0.9395029, 0.7544766, .9405936, 0.9402743,  0.9412337, 0.9415538,  0.941298, 0.9411695
#   0.9137742,0.9320154,0.9374562,0.9382258,0.9379061,0.9378424, 0.938674,0.9393139 ,0.9384821
#0.9176162,  0.9339352, 0.9372658, 0.9393123, 0.9395062,0.9393784,0.9399537,0.9388661,0.9395701

plot(numtree, e11, col =2, type='b',pch =19, xlab='number of trees', ylab = 'AUC of ROC')
lines(numtree, e12, col =3, type='b',pch =18)
lines(numtree, e13, col =4, type='b',pch =17)
lines(numtree, e14, col =6, type='b',pch =16)
legend(x="bottomright",legend=c("sqrt(p)","p/2","p/3","p"),col=c(2,3,4,6),lty=c(1,1,1,1),pch =c(19,18,17,16),cex=1.0)




### fit the random forrest or the bagging on the training set

numpred = ncol(data.train)-1
forest.fit <- randomForest(default_class ~ .,   data = defaultdata, subset = train, mtry = numpred, ntree = 800, importance = TRUE)
print(forest.fit) # view results 
importance(forest.fit) # importance of each predictor
#
### test the Bagging / Random forrest model on the small evaluation set
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
forest.perf = performance(pred,"tpr","fpr")
plot(forest.perf,col= 3,lwd=1)
#plot(bag.perf,col= 4,lwd=1, add=TRUE)
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)


################################################# Boosting model

## Devide the training set to 2 equal parts for k-fold cross validation
numfolds = 2
foldndata = numdata/numfolds

rowsmat = matrix(0, nrow = foldndata, ncol = numfolds)
totaldata = numdata
newrows = c(1:numdata)
for ( i in 1:numfolds)
{
  set.seed(2)
  vec = sample(1:totaldata, foldndata, replace=FALSE)
  rowsmat[, i] = vec
  newrows = newrows[-vec]
  totaldata = length(newrows)
}

## 2-fold cross validation on training set to pick the best number of trees for boosting
numtree = 5000
dep = c(4)
n = length(dep)
cverror = rep(0,n)##store the sensitivity accuracy
AUC = rep(0,n)
acc = rep(0, n)
for(r in c(1:n))
{
  kfoldcv.errorvec = rep(0,numfolds)
  AUCvec = rep(0,numfolds)
  accvec = rep(0,numfolds)
  for ( j in 1:numfolds)
  {
    indexvec = as.vector(rowsmat[, j])
    x.traincv = data.train[-indexvec, ]
    y.traincv = y.train[-indexvec]
    x.testcv = data.train[indexvec, ]
    y.testcv = y.train[indexvec]
   
    
    boost.fit = gbm((unclass(default_class)-1)~., data = x.traincv, distribution = 'bernoulli', n.tree = numtree, interaction.depth = dep[r])
    boost.pred = predict(boost.fit, newdata = x.testcv, n.tree = numtree, type = 'response')
    boost.predclass = rep("0", length(y.testcv))
    boost.predclass[boost.pred > 0.5]="1"
    tabl = table(boost.predclass, (unclass(y.testcv)-1))
    mean(boost.predclass==(unclass(y.testcv)-1))
    tp = tabl[2,2]
    fp = tabl[2,1]
    fn = tabl[1,2]
    tn = tabl[1,1]
    metricvec = metricfun (tp, fp, fn, tn)
    pred = prediction(boost.pred, y.testcv)
    boost.perf = performance(pred,"tpr","fpr")
    perf_auc =  performance(pred, measure = "auc")
    AUCvec[j] = as.numeric(perf_auc@y.values)
    kfoldcv.errorvec[j] = metricvec[2]
    accvec[j] = metricvec[1]
  } 
  cverror[r] = mean(kfoldcv.errorvec)
  AUC[r] = mean(AUCvec)
  acc[r] = mean(accvec)
}

print(cverror)
print(AUC)
print(acc)

numtree = c(1000, 3000, 5000)
#depthvec = c(2, 3, 4)
e12 = c(0.7999, 0.81685, 0.81735)# accuracy rate for depth =2
e22 = c( 0.7592516,  0.7655775,  0.770236)# auc of roc for depth =2
e32 = c( 0.9851665,  0.9519493, 0.9500874)#sensitivity for depth =2
e13 = c( 0.8019,0.8181,0.8179 )# accuracy rate for depth =3
e23 = c(0.7623014,0.7688349, 0.7728328)# auc of roc for depth =3
e33 = c( 0.984397,0.9493148, 0.9477737)#sensitivity for depth =3
e14 = c(0.80285,  0.8182, 0.8178)# accuracy rate for depth =4
e24 = c(0.7640859, 0.77072, 0.7739998)# auc of roc for depth =4
e34 = c(0.9840762,0.9494433, 0.9475175)#sensitivity for depth =4

### plot the accuracy rate vs num trees
plot(numtree, e13, col = 2, type = 'b', pch = 19, xlab = 'number of trees', ylab = 'Accuracy rate',ylim = c(0.799, 0.82))
lines(numtree, e14, col = 4, type = 'b', pch = 15)
lines(numtree, e12, col = 3, type = 'b', pch = 17)
legend(x="topleft",legend=c("d = 3","d = 4","d = 2"),col=c(2,4,3),lty=c(1,1,1),pch =c(19,15,17),cex=1.0)



### plot the  auc vs num trees
plot(numtree, e23, col = 2, type = 'b', pch = 19, xlab = 'number of trees', ylab = 'AUC OF ROC', ylim = c(0.75, 0.78))
lines(numtree, e24, col = 4, type = 'b', pch = 15)
lines(numtree, e22, col = 3, type = 'b', pch = 17)
legend(x="topleft",legend=c("d = 3","d = 4","d = 2"),col=c(2,4,3),lty=c(1,1,1),pch =c(19,15,17),cex=1.0)



###fit the bossting model on the training data
boost.fit = gbm((unclass(default_class)-1)~., data = data.train, distribution = 'bernoulli', n.tree = 5000, interaction.depth = 4)

### test the Boosting model on the small evaluation set
boost.pred = predict(boost.fit, newdata = eval.data, n.tree = 5000, type = 'response')
boost.predclass = rep("0", length(y.eval.data))
boost.predclass[boost.pred > 0.2]="1"
tabl = table(boost.predclass, (unclass(y.eval.data)-1))
mean(boost.predclass==(unclass(y.eval.data)-1))
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)

boost.pred = prediction(boost.pred, y.eval.data)
boost.perf = performance(boost.pred,"tpr","fpr")
plot(boost.perf,col=4,lwd=1,main="ROC Curve for the Boosting model")
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(boost.pred, measure = "auc")
print(perf_auc@y.values)

points( 1-0.7653127,0.6681514, pch = 19 ,col = 'red')

### test the Boosting model on the test set with aternaive cutoff
boost.pred = predict(boost.fit, newdata = data.test, n.tree = 5000, type = 'response')
boost.predclass = rep("0", length(y.test))
boost.predclass[boost.pred > 0.15]="1"
tabl = table(boost.predclass, (unclass(y.test)-1))
mean(boost.predclass==(unclass(y.test)-1))
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)



