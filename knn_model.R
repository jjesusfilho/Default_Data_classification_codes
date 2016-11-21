## Clear memmory
rm(list=ls(all=TRUE))

## loading required packages
library(readxl)
library(FNN)
library('ROCR')


## loading required functions
source('metrics.R')

### Read the data from Excel file
defaultdata <- read_excel("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients.xlsx")
#View(defaultdata)
#df = read.xls("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients_old.xls", sheet = 1, header = TRUE)
defaultdata = data.frame(defaultdata[,-1])



## convert the catogorical variales to predictors
defaultdata$default_class = as.factor(defaultdata$default_class)
defaultdata$SEX = factor(defaultdata$SEX)
defaultdata$EDUCATION = factor(defaultdata$EDUCATION)
defaultdata$MARRIAGE = factor(defaultdata$MARRIAGE)



# validation set approach
# create a training set that randomly samples from 2/3 of the available data 
#set.seed(10)
numdata = 2*nrow(defaultdata)/3
train = sample(1:nrow(defaultdata), numdata)

# the test rows are those not in train
test = (-train)

# test data
y.test = defaultdata$default_class[test]
x.test = model.matrix(default_class ~ ., defaultdata[-train,])[,-1]


## training data
y.train = defaultdata$default_class[train]
x.train = model.matrix(default_class ~ ., defaultdata[train,])[,-1]

# set aside a  small evaluation set from the test data to be used for
#developing post-processing techniques, such as alternative probabilitycutoffs (numdata = 2000)
set.seed(1)
numdataeval = nrow(x.test)/5
eval = sample(1:nrow(x.test), numdataeval)
eval.data = x.test[eval, ]
y.eval.data = y.test[eval]
print(sum(y.eval.data==0)/length(y.eval.data)) ### No information accuracy on the small evaluation set


### LOOCV on training set to pick the best k value
kvec = c( 80, 100, 150, 200)
n = length(kvec)
cv.error=rep(0,n)
for (i in c(1:length(kvec))){
  knn.pred = knn.cv(x.train, y.train, k = kvec[i], prob = FALSE)
  cv.error[i] = mean(knn.pred == y.train)
}
plot(kvec, cv.error, col = 'red', type = 'b', pch = 19)

## k-fold cross validation on training set to pick the best k value
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

kvec = c(60, 70, 100, 120, 150)
n = length(kvec)
cverror = rep(0,n)
AUC = rep(0,n)
for(r in c(1:n))
{
kfoldcv.errorvec = rep(0,numfolds)
AUCvec = rep(0,numfolds)
for ( j in 1:numfolds)
{
  indexvec = as.vector(rowsmat[, j])
  x.traincv = x.train[-indexvec, ]
  y.traincv = y.train[-indexvec]
  x.testcv = x.train[indexvec, ]
  y.testcv = y.train[indexvec]
  knn.pred = knn(x.traincv, x.testcv, y.traincv, k = kvec[r], prob = TRUE)
  knn.prob = attributes(knn.pred)$prob
  knn.prob[which(knn.pred==0)]=1-knn.prob[which(knn.pred==0)]
  pred = prediction(knn.prob,y.testcv)
  perf_auc =  performance(pred, measure = "auc")
  AUCvec[j] = as.numeric(perf_auc@y.values)
  kfoldcv.errorvec[j] = mean(knn.pred == y.testcv)
} 
cverror[r] = mean(kfoldcv.errorvec)
AUC[r] = mean(AUCvec)
}

plot(kvec, cverror, col = 'red', type = 'b', pch = 19, xlab = 'number of neighbours-k', ylab = 'classification accuracy')
#plot(kvec, AUC, col = 'green', type = 'b', pch = 17, xlab = 'number of neighbours-k', ylab = 'AUC of ROC')
  



### fit knn model on the training data and do the predictions on small evaluation test
knn.pred =class::knn(x.train, eval.data, y.train, k=100, prob=TRUE)
knn.prob = attributes(knn.pred)$prob
## knn.prob <- attr(knn.pred, "prob", exact=TRUE)
tabl = table(knn.pred, y.eval.data)
mean(knn.pred == y.eval.data)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)


###plot the ROC curve
knn.prob[which(knn.pred==0)]=1-knn.prob[which(knn.pred==0)]
pred = prediction(knn.prob,y.eval.data)
perf = performance(pred,"tpr", "fpr")
plot(perf,col='orange',lwd=1, add = TRUE)
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)

plot(perf,col='orange',lwd=1)
abline(a=0,b=1,lty=5,col="Gray")
legend(x="bottomright",legend=c("Decision tree","Bagging","Random forest","Boosting", "logistic regression", "Naive Bayesian", "KNN"),col=c(2,4,3,6, 25, 21, 'orange' ),lty=c(1,1,1,1,1,1,1),cex=1.0)



### knn model on the validation test set
knn.pred =class::knn(x.train, x.test, y.train, k=100, prob=TRUE)
knn.prob = attributes(knn.pred)$prob
## knn.prob <- attr(knn.pred, "prob", exact=TRUE)
tabl = table(knn.pred, y.test)
mean(knn.pred == y.test)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)


###plot the ROC curve
knn.prob[which(knn.pred==0)]=1-knn.prob[which(knn.pred==0)]
pred = prediction(knn.prob,y.test)
perf = performance(pred,"tpr", "fpr")
plot(perf,col="Blue",lwd=1,xlab="False positive rate", ylab="True positive rate",main="ROC Curve")
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)

