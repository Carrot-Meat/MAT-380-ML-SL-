setwd("C:/Users/Me/Desktop/R")

require(caTools)

#Initial Data Processing
Quasar <- read.table("SDSS_train2.txt", header = TRUE)
Quasar <- Quasar[complete.cases(Quasar), ]
Quasar[,34] <- as.numeric(Quasar[,34])-1

set.seed(314)
split <- sample.split(Quasar$ID, SplitRatio = .70)
train <- subset(Quasar, split == TRUE)
test <- subset(Quasar, split == FALSE)

X2 <- data.matrix(train[c(2:23)])
X2t <- data.matrix(test[c(2:23)])
maxs <- apply(X2, 2, max)
mins <- apply(X2, 2, min)
X2 <- (scale(X2,center = mins, scale = maxs - mins))
X2t <- (scale(X2t,center = mins, scale = maxs - mins))
Y2 <- train$radio_loud
Y2t <- test$radio_loud

#MLR ---------------------------------------------------------
require(mlr)
require(deepnet)

X2<- as.data.frame(X2)
data2f <- as.data.frame(cbind(Y2,X2))
data2f0 <- data2f[which(data2f$Y2=='0'),]
data2f1 <- data2f[which(data2f$Y2=='1'),]                        
X2f = rbind(
  data.frame(data2f0, class = "A"),
  data.frame(data2f1, class = "B"))

task = makeClassifTask(data = X2f, target = "class")
task.smote = smote(task, rate = 9.75, nn = 5)

set.seed(3141)
lrn <- makeLearner("classif.dbnDNN", predict.type = "response", activationfun = "sigm",
                   learningrate = 0.001,  momentum = 0.5, learningrate_scale = 1, output = "softmax", 
                   numepochs = 100,  batchsize = 100, cd = 1)
rdesc = makeResampleDesc("CV", iters = 10)
r = resample("classif.dbnDNN", task.smote, rdesc, measure = acc)


ctrl = makeTuneControlGrid()
res <- tuneParams("classif.dbnDNN", task = task.smote, resampling = rdesc, par.set = num_ps, control = ctrl,
                  measure = acc)

lrnnn <- makeLearner("classif.dbnDNN", predict.type = "response",
                     learningrate = 0.001, 
                     numepochs = 100)
lrnn = setHyperPars(makeLearner("classif.dbnDNN",predict.type = "response"), par.vals = res$x)
m = mlr::train(lrn,task.smote)
pred.smote1 = predict(m, task = task.smote)
performance(pred.smote1,measures = acc)

X2t<- as.data.frame(X2t)
data2ft <- as.data.frame(cbind(Y2t,X2t))
data2f0t <- data2f[which(data2ft$Y2=='0'),]
data2f1t <- data2f[which(data2ft$Y2=='1'),]                        
X2ft = rbind(
  data.frame(data2f0t, class = "A"),
  data.frame(data2f1t, class = "B"))

taskt = makeClassifTask(data = X2ft, target = "class")
pred.smote2 = predict(mod.smote, task = taskt)
performance(pred.smote2,measures = acc)




#SVM --------------------------
num_ps <- makeParamSet(
  makeDiscreteParam("learningrate", c(0.0001,0.001,0.01,0.1)),
  makeDiscreteParam("numepochs", c(32,64,128,256)),
  makeDiscreteParam("hidden_dropout", c(0.2, 0.4, 0.6, 0.8))
) 
lrnsvm <- makeLearner("classif.ksvm", cross = 10L)
mod.svm = mlr::train(lrnsvm, task)
pred.svm1 = predict(mod.svm, task = task)
performance(pred.smote1,measures = acc)
