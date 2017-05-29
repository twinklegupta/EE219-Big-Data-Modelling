
################################################################################
initial_model <- lm(Size.of.Backup..GB. ~ Week.. + Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID+File.Name+Backup.Time..hour.,network_backup_dataset)
summary(initial_model)
rmse_initial <- sqrt(sum((network_backup_dataset$Size.of.Backup..GB.-initial_model$fitted.values)^2)/nrow(network_backup_dataset))
# higher the t-value, better the parameter. Therefore we choose the best 3 parameters : Backuptime, days, startTime
regression_model <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,network_backup_dataset)
summary(regression_model)
rmse_main <- sqrt(sum((network_backup_dataset$Size.of.Backup..GB.-regression_model$fitted.values)^2)/nrow(network_backup_dataset))
plot(regression_model)
#RMSE = 0.07404
#To cross validate, we sampled the data and randomized it
shuffled_data<-network_backup_dataset[sample(nrow(network_backup_dataset)),]
#created 10 folds
folds <- cut(seq(1,nrow(shuffled_data)),breaks=10,labels=FALSE)
predicted_values = matrix(data=NA,nrow=nrow(network_backup_dataset),ncol=1)
#Perform 10 fold cross validation
for(i in 1:10){
  #separate test and trainign data using which 
  test_indices <- which(folds==i,arr.ind=TRUE)
  testData <- shuffled_data[test_indices, ]
  trainData <- shuffled_data[-test_indices, ]
  test_model <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,trainData) #Train the Model on 90% of the data
  test_prediction <- predict(test_model,testData) #Test on the% of the data
  row_names <- rownames(as.matrix(test_prediction))
  for(j in 1:length(test_prediction))
  {
    predicted_values[as.numeric(row_names[j])] = test_prediction[j] #Store the predicted values
  }
}
actual_values <- network_backup_dataset$Size.of.Backup..GB.
cv_rmse <- sqrt(sum((actual_values-predicted_values)^2)/nrow(network_backup_dataset)) #Calculate the RMSE = 0.7409
################################################################################
#Fitted Values and Actual Values Scatter Plot over Time
library(lattice)
dayList <- levels(network_backup_dataset$Day.of.Week)[c(2,6,7,5,1,3,4)]
numeric_day <- as.numeric(factor(network_backup_dataset$Day.of.Week,dayList))
network_backup_dataset$numeric.day <- numeric_day
res <- matrix(network_backup_dataset$numeric.day,nrow = length(network_backup_dataset$numeric.day))
res <- cbind(res, x2 = network_backup_dataset$Backup.Start.Time...Hour.of.Day)
res <- cbind(res, x3 = network_backup_dataset$Backup.Time..hour.)
res <- cbind(res, Observed = network_backup_dataset$Size.of.Backup..GB. )
res <- cbind(res, Predicted = predicted_values)
res <- data.frame(res)
xyplot(c(res$Observed,as.double(res$V5)) ~ res$V1, data = res,  auto.key = TRUE,xlab='Week',ylab='Observed Backup Size in GB')
xyplot(res$V5 ~ res$V1, data = res,  auto.key = TRUE,xlab='Week',ylab='Backup Size in GB')
#xyplot(network_backup_dataset$Size.of.Backup..GB. ~ res[,1], data = res, type = c("p","r"), col.line = "red")
plot(res$V1,res$Observed)
plot(res$V1, res$V5,xlab='Week',col = 4)
points(res$V1,res$Observed, col = 2)
################################################################################
#Residuals versus Fitted Values Plot
library('ggplot2')
qplot(y=(actual_values-predicted_values),x=predicted_values,xlab='Fitted Values',ylab='Residuals')
################################################################################
#Cross Validation using in-built function
library('DAAG')
cv_model<-cv.lm(data = network_backup_dataset, form.lm = formula
                (Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.),
                m = 10)
predicted_values <- cv_model$cvpred
actual_values <- network_backup_dataset$Size.of.Backup..GB.
cv_model_inbuilt_rmse <- sqrt(sum((actual_values-predicted_values)^2)/nrow(network_backup_dataset)) #RMSE = 07407

