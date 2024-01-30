library(MASS)
library(pROC)
library(nnet)
library(car)
library(glmnet)
library(randomForest)

set.seed(67982193)
options(max.print=100)
wine <- read.csv("winequality-white.csv", sep=";")
dim(wine)

summary(wine)

wine$quality <- ifelse(wine$quality>6,"good","bad")
wine$quality <- as.factor(wine$quality)

per <- sample(x = nrow(wine))
wine.train <- wine[which(per <= 4098),]
wine.test <- wine[which(per > 4098),]
rescale <- function(x1, x2) {   # rescaling both sets
  for (col in 1:ncol(x1)) {
    a <- min(x2[, col])
    b <- max(x2[, col])
    x1[, col] <- (x1[, col] - a) / (b - a)}
  x1}
wine.train.resc <- data.frame(cbind(rescale(wine.train[,-12], wine.train[,-12]),
                                    quality = wine.train$quality))
wine.test.resc <- data.frame(cbind(rescale(wine.test[,-12], wine.train[,-12]),
                                   quality = wine.test$quality))

options(max.print=10)

fit.rf <- randomForest(quality ~., data = wine.train)   # training via RF method
pred.train.rf <- predict(fit.rf, newdata = wine.train[, -12])   # prediction on training set
pred.test.rf <- predict(fit.rf, newdata = wine.test[, -12])   # prediction on test set

(pred.test.rf.prob <- predict(fit.rf, newdata = wine.test[, -12], type = "prob"))

fit.rf82 <- randomForest(quality ~., data = wine.train, cutoff = c(0.8, 0.2))
pred.test.rf82 <- predict(fit.rf82, newdata = wine.test[, -12])
(rf_table82 <- table(pred.test.rf82, wine.test$quality, dnn = c("Prediction", "Observation")))

logit.cv.lr <- cv.glmnet( x = as.matrix(wine.train.resc[, 1:11]), y = wine.train.resc[, 12],
                          family = "multinomial")   # training via LR
lascv.pred.train.lr <- predict(object = logit.cv.lr, type = "class", s = logit.cv.lr$lambda.min, 
                               newx = as.matrix(wine.train.resc[, 1:11]))
lascv.pred.test.lr <- predict(logit.cv.lr, type = "class", s = logit.cv.lr$lambda.min, 
                              newx = as.matrix(wine.test.resc[, 1:11]))

(lascvmis.train.lr <- mean(ifelse(lascv.pred.train.lr == wine.train$quality, yes = 0, no = 1)))
(lascvmis.test.lr <- mean(ifelse(lascv.pred.test.lr == wine.test$quality, yes = 0, no = 1)))

table(lascv.pred.test.lr, wine.test$quality, dnn = c("Prediction", "Observation"))

(lascv.pred.test.lr.prob <- predict( logit.cv.lr, type = "response",
                                     s = logit.cv.lr$lambda.min, newx = as.matrix(wine.test.resc[, 1:11]))[,,1])

ROC_lr_las <- roc(wine.test$quality, lascv.pred.test.lr.prob[,2])
ROC_rf <- roc(as.numeric(wine.test$quality), as.numeric(pred.test.rf.prob[,2]))

plot(ROC_lr_las, col = "red", main = "ROC curve", xlim = c(1,0), legacy.axes = TRUE)
lines(ROC_rf, col = "blue")
legend(x = "topleft", legend = c("LASSO_LR", "RF"), lty = c(1, 1), col = c("red", "blue"), lwd = 2)

(ROC_lr_las_auc <- auc(ROC_lr_las))
(ROC_rf_auc <- auc(ROC_rf))
