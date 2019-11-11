setwd("C:/Users/Me/Desktop/R")

require(mlr)
require(deepnet)
require(nnet)
require(caret)
require(caTools)
require(kernlab)
require(MASS)
#Data Initialization and Prep -------------------------------
Quasar <- read.table("SDSS_train2.txt", header = TRUE)
Quasar <- Quasar[complete.cases(Quasar), ]
Quasar <- Quasar[,c(4:24,34)]
Quasart <- read.table("SDSS_test.txt", header = TRUE)
Quasart <- Quasart[complete.cases(Quasart), ]
Quasart <- Quasart[,c(4:24,34)]
X2 <- data.frame(Quasar[,2:21])
maxs <- apply(X2, 2, max)
mins <- apply(X2, 2, min)
X2 <- (scale(X2,center = mins, scale = maxs - mins))
X2L <- data.frame(Quasar[,c(2,4,6,8,10,12:21)])
X2t <- data.frame(Quasart[,2:21])
maxs <- apply(X2t, 2, max)
mins <- apply(X2t, 2, min)
X2t <- (scale(X2t,center = mins, scale = maxs - mins))
X2Lt <- data.frame(Quasart[,c(2,4,6,8,10,12:21)])
Y2 <- data.frame(Quasar$radio_loud)
Y2t <- data.frame(Quasart$radio_loud)
data <- (cbind(X2,Y2))
dataL <- (cbind(X2L,Y2))
datat <- (cbind(X2t,Y2t))
datatL <- (cbind(X2Lt,Y2t))
colnames(data)[21]<-"class"
colnames(datat)[21] <- "class"
colnames(dataL)[16] <- "class"
colnames(datatL)[16] <- "class"
task = makeClassifTask(data = datat, target = "class")
taskL = makeClassifTask(data = dataL, target = "class")
tasktL = makeClassifTask(data = datatL, target = "class")

set.seed(314)
lrn <- makeLearner("classif.ksvm", predict.type = "response", type = "nu-svc", kernel ="rbfdot")
lrn.smote <- makeSMOTEWrapper(lrn, sw.rate=10, sw.nn = 5)
mod.smote <- mlr::train(lrn.smote, task)


set.seed(314)
lrn1 <- makeLearner("classif.lda",predict.type = "response")
lrn.smote1 <- makeSMOTEWrapper(lrn1, sw.rate=10, sw.nn = 5)
mod.smote1 <- mlr::train(lrn.smote1,taskL)

set.seed(314)
lrn2 <- makeLearner("classif.nnet",predict.type = "response", par.vals = list(maxit = 10000, size = 14, abstol = 0.0001))
lrn.smote2 <- makeSMOTEWrapper(lrn2, sw.rate=10, sw.nn = 5)
mod.smote2 <- mlr::train(lrn.smote2,task)

set.seed(123)
lrn3 <- makeLearner("classif.dbnDNN", predict.type = "response",hidden = c(16,8), activationfun = "sigm",
                           learningrate = 0.001,  momentum = 0.5, learningrate_scale = 1, output = "softmax", 
                           numepochs = 100,  batchsize = 32, cd = 1)
lrn4 <- makePreprocWrapperCaret(lrn3, ppc.center = TRUE, ppc.scale = TRUE, measures = tnr)
lrn.smote3 <- makeSMOTEWrapper(lrn4, sw.rate=9.75, sw.nn = 5)
mod.smote3 <- mlr::train(lrn.smote3,task)
remove(mod.smote3)


pred.smote3 = predict(mod.smote3, task = taskt)
performance(pred.smote3, measures = list(acc,npv))

taskt = makeClassifTask(data = datat, target = "class")
pred.smote2 = predict(mod.smote2, task = taskt)
performance(pred.smote2, measures = list(acc,npv))
t <- calculateConfusionMatrix(pred.smote)
t

pred.smote1 = predict(mod.smote1, task = tasktL)
performance(pred.smote1, measures = list(acc,npv))
t <- calculateConfusionMatrix(pred.smote1)
t
