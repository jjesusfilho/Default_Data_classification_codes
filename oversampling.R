### Implement the Oversampling method to get a balanced data set



## Clear memmory
rm(list=ls(all=TRUE))

## loading required packages
library(readxl)
library(FNN)
library(ROCR)
library(FNN)

### Read the data from Excel file
defaultdata <- read_excel("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients.xlsx")
#View(defaultdata)
#df = read.xls("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients_old.xls", sheet = 1, header = TRUE)
defaultdata = data.frame(defaultdata[,-1])



## convert the catogorical variales to predictors
defaultdataold <- defaultdata
defaultdataold$default_class = as.factor(defaultdata$default_class)
defaultdataold$SEX = factor(defaultdata$SEX)
defaultdataold$EDUCATION = factor(defaultdata$EDUCATION)
defaultdataold$MARRIAGE = factor(defaultdata$MARRIAGE)


# validation set approach
# create a training set that randomly samples from 2/3 of the available data 
set.seed(10)
numdata = 2*nrow(defaultdataold)/3
train = sample(1:nrow(defaultdataold), numdata)

# the test rows are those not in train
test = (-train)


## training data
set.seed(3)
traindata = defaultdataold[train,]
traindatacl2 = traindata[which(traindata$default_class=="1"), ]
traindatacl1 = traindata[-which(traindata$default_class=="1"), ]

numtraindatacl2 = nrow(traindatacl2)
bootstaprows = sample(1:numtraindatacl2, nrow(traindatacl1), replace = TRUE)
addrows = traindatacl2[bootstaprows,]
newtraindata = rbind(traindata, addrows)
y.train = newtraindata$default_class
x.train = model.matrix(default_class ~ ., newtraindata)[,-1]

# test data
y.test = defaultdataold$default_class[test]
x.test = model.matrix(default_class ~ ., defaultdataold[-train,])[,-1]

##um(newtraindata$default_class==0)/nrow(newtraindata)
##sum(defaultdataold$default_class==0)/nrow(defaultdataold)
##sum(traindata$default_class==0)/nrow(traindata)



## k-fold cross validation on training set to pick the best k value
numfolds = 10
foldndata = floor(nrow(newtraindata)/numfolds)

rowsmat = matrix(0, nrow = foldndata, ncol = (numfolds-1))
totaldata = nrow(newtraindata) 
newrows = c(1:nrow(newtraindata))
for ( i in 1:(numfolds-1))
{
  vec = sample(1:totaldata, foldndata, replace=FALSE)
  rowsmat[, i] = vec
  newrows = newrows[-vec]
  totaldata = length(newrows)
}
last.trainset = newrows


kvec = c(1, 2, 5, 10, 50)
n = length(kvec)
cverror = rep(0,n)
for(r in c(1:n))
{
  kfoldcv.errorvec = rep(0,numfolds)
  for ( j in 1:numfolds)
  {
    if (j == 10) {
    indexvec = last.trainset
    } else {
    indexvec = as.vector(rowsmat[, j])
    }
    x.traincv = x.train[-indexvec, ]
    y.traincv = y.train[-indexvec]
    x.testcv = x.train[indexvec, ]
    y.testcv = y.train[indexvec]
    knn.pred = knn(x.traincv, x.testcv, y.traincv, k = kvec[r], prob = FALSE)
    kfoldcv.errorvec[j] = mean(knn.pred == y.testcv)
 } 
  cverror[r] = mean(kfoldcv.errorvec)
}
plot(kvec, cverror, col = 'red', type = 'b', pch = 19, xlab = 'number of neighbours-k', ylab = 'classification accuracy')




### knn model
knn.pred = knn(x.train, x.test, y.train, k=1, prob=TRUE)
knn.prob = attributes(knn.pred)$prob
## knn.prob <- attr(knn.pred, "prob", exact=TRUE)
table(knn.pred, y.test)
mean(knn.pred == y.test)

##plot the ROC curve
knn.prob[which(knn.pred==0)]=1-knn.prob[which(knn.pred==0)]
pred = prediction(knn.prob,y.test)
perf = performance(pred,"tpr", "fpr")
plot(perf,col="Blue",lwd=1,xlab="False positive rate", ylab="True positive rate",main="ROC Curve")
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)
# 0.6575428 for k = 100

## To understand the ROC curve
## naive prob of defaulting (using thhe fractions from the training data)
prdef = sum(defaultdataold$default_class==1)/nrow(defaultdataold)
prntdef = 1-prdef
naive.prob = rep(prdef, length(y.test))

pred = prediction(naive.prob,y.test)
perf = performance(pred,"tpr", "fpr")
plot(perf,col="Blue",lwd=1,xlab="False positive rate", ylab="True positive rate",main="ROC Curve")
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)

######################## linear model tests
x.testnew = defaultdataold[-train,]
x.trainnew = newtraindata

## LDA model
lda.fit = lda(default_class ~ ., data = newtraindata)
lda.pred = predict(lda.fit, x.testnew)
lda.class = lda.pred$class
table(lda.class, y.test)
mean(lda.class == y.test)

# to create ROC 
pred = prediction(lda.pred$posterior[,2],as.numeric(y.test))
perf = performance(pred,"tpr","fpr" )
plot(perf,col="Blue",lwd=1,xlab="False positive rate",ylab="True positive rate",main="ROC Curve" )
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)


### logistic regression models
glm.fit = glm(default_class ~ ., data = newtraindata, family=binomial)
##glm.fit = glm(default_class ~ LIMIT_BAL+EDUCATION+MARRIAGE+PAY_0+BILL_AMT1+PAY_AMT1+PAY_AMT2+PAY_AMT6, data = newtraindata, family=binomial)

## compute the test set error
glm.probtest = predict(glm.fit, x.testnew, type='response')
glm.predtest = rep("0", length(y.test))
glm.predtest[glm.probtest > 0.2]="1"
tabl = table(glm.predtest, y.test)
mean(glm.predtest == y.test)
precis = tabl[1,2]/sum(tabl[1,])
print(precis) ###  the fraction of people defaulted out of people that we predicted will not default

# to create ROC 
pred = prediction(glm.probtest,as.numeric(y.test))
perf = performance(pred,"tpr","fpr")
plot(perf,col="Blue",lwd=1,xlab="False positive rate",ylab="True positive rate",main="ROC Curve" )
abline(a=0,b=1,lty=5,col="Gray")

## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)


