### Decision tree with rpart on a balanced data set



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



## convert the catogorical variales to predictors
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

## training data
set.seed(3)
traindata = defaultdata[train,]
traindatacl2 = traindata[which(traindata$default_class=="yes"), ]
traindatacl1 = traindata[which(traindata$default_class=="No"), ]

numtraindatacl2 = nrow(traindatacl2)
set.seed(1000)
bootstaprows = sample(1:numtraindatacl2, nrow(traindatacl1)-5000, replace = TRUE)
addrows = traindatacl2[bootstaprows,]
newtraindata = rbind(traindata, addrows)


y.train = newtraindata$default_class
y.test = defaultdata$default_class[test]

data.test = defaultdata[-train,]
data.train= newtraindata


######################## Decision tree

### fit the tree on training data
mycontrol = rpart.control(cp = 0.00, xval = 10)
tree.fit = rpart(default_class~., data = data.train, method = 'class', control = mycontrol)
pruned.fit<- prune(tree.fit, cp = tree.fit$cptable[which.min(tree.fit$cptable[,"xerror"]),"CP"])
print(pruned.fit)
#plot(pruned.fit)
#text(pruned.fit)

## confusion matrix on the validation test set
tree.pred = predict(pruned.fit, newdata = data.test,type = 'cl')
tabl = table(tree.pred, y.test)
mean(tree.pred==y.test)
tp = tabl[2,2]
fp = tabl[2,1]
fn = tabl[1,2]
tn = tabl[1,1]
metricvec = metricfun (tp, fp, fn, tn)
print(metricvec)

print(tree.fit$cptable)
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



######################## Random forest model


numpred = ncol(data.train)-1
forest.fit <- randomForest(default_class ~ .,   data = defaultdata, subset = train, mtry = sqrt(numpred), ntree = 700, importance = TRUE)

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
# 23.44%, 18.95% , 18.59%, 18.33%, 18.55%,  18.37%

