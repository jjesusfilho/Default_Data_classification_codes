## Clear memmory
rm(list=ls(all=TRUE))

## loading required packages
library(readxl)
library(glmnet)
library(MASS)
library(ROCR)

## loading required functions
source('metrics.R')

### Read the data from Excel file
defaultdata <- read_excel("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients.xlsx")
#View(defaultdata)
#df = read.xls("C:/Users/Madushani/Google Drive/Insight_data_anlysis/default_of_credit_card_clients_old.xls", sheet = 1, header = TRUE)
defaultdata = data.frame(defaultdata[,-1])


## convert the catogorical variales to factors
defaultdata$default_class = as.factor(defaultdata$default_class)
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
    
    glm.fit = glm(default_class ~ LIMIT_BAL + SEX+AGE+ EDUCATION+MARRIAGE+BILL_AMT1+PAY_0+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6, data = x.traincv, family=binomial)
    glm.probtest = predict(glm.fit, newdata = x.testcv, type='response')
    glm.predtest = rep(0, length(y.testcv))
    glm.predtest[glm.probtest > 0.5]=1
    tabl = table(glm.predtest, y.testcv) 
    mean(glm.predtest==y.testcv)
    tp = tabl[2,2]
    fp = tabl[2,1]
    fn = tabl[1,2]
    tn = tabl[1,1]
    metricvec = metricfun (tp, fp, fn, tn)
    pred = prediction(glm.predtest, y.testcv)
    perf_auc =  performance(pred, measure = "auc")
    AUCvec[j] = as.numeric(perf_auc@y.values)
    kfoldcv.errorvec[j] = metricvec[2]
    accvec[j] = metricvec[1]
  } 
mean(kfoldcv.errorvec)
mean(AUCvec)
mean(accvec)

## full model, LIMIT_BAL+PAY_0, PAY_O, PAY_2+PAY_0, with no correlated predictors I.E, LIMIT_BAL + SEX+AGE+ EDUCATION+MARRIAGE+BILL_AMT1+PAY_0+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6
e1 = c(0.81025, 0.8136, 0.8216, 0.8112,  0.81195)#accuracy rate
e2 = c(0.6016996, 0.6127626,  0.6445972, 0.6033556, .605831)#auc
e3 = c( 0.9729185, 0.9701592,  0.959534,  0.9732389, 0.9726689)#sensitivity
x = c(1,2,3,4,5)
plot(x, e1, type='b', col = 'blue',pch = 16, xlab='Model number', ylab = 'accuracy rate')
plot(x, e2, type='b', col = 'red',pch = 15, xlab='Model number', ylab = 'AUC of ROC')



### Fit the model on the training set
### logistic regression models
glm.fit = glm(default_class ~ PAY_0, data = defaultdata, family=binomial, subset = train)
summary(glm.fit)


## test the model on the small evaluation set
glm.probtest = predict(glm.fit, eval.data, type='response')
glm.predtest = rep("0", length(y.eval.data))
glm.predtest[glm.probtest > 0.3]="1"
tabl = table(glm.predtest, as.factor((unclass(y.eval.data)-1)))
mean(glm.predtest == as.factor((unclass(y.eval.data)-1)))
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)


# to create ROC 
log.pred = prediction(glm.probtest,as.factor((unclass(y.eval.data)-1)))
log.perf = performance(log.pred,"tpr","fpr" )
plot(log.perf,col=25,lwd=1, add = TRUE)
abline(a=0,b=1,lty=5,col="Gray")


## compute the area under the ROC curve
perf_auc = performance(pred, measure = "auc")
print(perf_auc@y.values)

