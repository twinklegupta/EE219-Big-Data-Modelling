#Problem4

colnames(housing_data) <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV");


## Linear regression

library('usdm')

#choosing the best parameters
#show summary in report
#explain significance of t-value
model1 <- lm(MEDV~.,data=housing_data)
sm1 <- summary(model1)
rmse1 <- sqrt(sum((housing_data$MEDV-model1$fitted.values)^2)/nrow(housing_data))

#taking parameters with highest t-value (shown with 3 starts in summary)-> these are ZN,NOX,RM,DIS,RAD,PTRATIO,B,LSTAT
model2 <- lm(MEDV ~ ZN+NOX+RM+DIS+RAD+PTRATIO+B+LSTAT,data=housing_data)
sm2 <- summary(model2)
rmse2 <- sqrt(sum((housing_data$MEDV-model2$fitted.values)^2)/nrow(housing_data))

#fine tuning the model further-> taking only most important parameters from model2-> NOX,RM,DIS,PTRATIO,B,LSTAT
model3 <- lm(MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT,data=housing_data)
sm3 <- summary(model3)
rmse3 <- sqrt(sum((housing_data$MEDV-model3$fitted.values)^2)/nrow(housing_data))

################################################################################
#cross validation
sampledData<-housing_data[sample(nrow(housing_data)),]

#creating 10 folds
folds <- cut(seq(1,nrow(sampledData)),breaks=10,labels=FALSE)
fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
residual_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
actual_values <- housing_data$MEDV
#Performing 10 fold cross validation
for(fold in 1:10){
 
  test_indices <- which(folds==fold,arr.ind=TRUE)
  testData <- sampledData[test_indices, ]
  trainData <- sampledData[-test_indices, ]
  tempModel <- lm(MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT,trainData) 
  predicted_values <- predict(tempModel,testData) 
  rows <- rownames(as.matrix(predicted_values))
  
  for(j in 1:length(predicted_values))
  {
    fitted_values[as.numeric(rows[j])] = predicted_values[j] 
    #residual_values[as.numeric(rows[j])] = actual_values[as.numeric(rows[j])] - predicted_values[j] 
  }
}

rmse <- sqrt(sum((actual_values-fitted_values)^2)/nrow(housing_data)) # RMSE
plot(tempModel)

# plot residual values vs fitted values
qplot(y=(actual_values-fitted_values),x=fitted_values,xlab='Fitted Values',ylab='Residuals')
qplot(y = fitted_values, x = actual_values, xlab = "Actual Values", ylab = "Fitted Values")

library('DAAG')
cv_model<-cv.lm(data = housing_data, form.lm = formula
                (MEDV ~ NOX+RM+DIS+PTRATIO+B+LSTAT),
                m = 10) # plots predicted vs actual values for every fold internally, so just used it off the shelf

plot(x = fitted_values, y = actual_values)


###########################################################################################
#                                     POLYNOMIAL REGRESSION                               #
###########################################################################################


shuffled_data<-housing_data[sample(nrow(housing_data)),] #Randomly distributing the data

#Create 10 equally size folds
folds <- cut(seq(1,nrow(shuffled_data)),breaks=10,labels=FALSE)


all_rmse = NULL

#10 fold cross validation
for(deg in 1:10)
{
  fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
  rmse = NULL
  for(fold in 1:10){
    
    test_indices <- which(folds==fold,arr.ind=TRUE)
    trainData <- (shuffled_data[-test_indices, ])
    pr_model <- lm(MEDV ~ polym(NOX,RM,DIS,PTRATIO,B,LSTAT,degree = deg,raw = TRUE),data=trainData)
    testData <- shuffled_data[test_indices, ]
    predicted_values <- predict(pr_model,testData)
    name <- rownames(as.matrix(predicted_values))
    for(j in 1:length(predicted_values))
    {
      fitted_values[as.numeric(name[j])] = predicted_values[j] 
    }
  }
  actual_values <- housing_data$MEDV
  rmse <- sqrt(mean((actual_values-fitted_values)^2))
  all_rmse <- append(all_rmse,rmse)
}
#best degree = 2
plot(all_rmse[1:5],xlab="Degree of Polynomial",ylab="RMSE")
plot(all_rmse,xlab="Degree of Polynomial",ylab="RMSE")
plot(all_rmse[1:4],xlab="Degree of Polynomial",ylab="RMSE",type="l") #use this for report
plot(all_rmse,xlab="Degree of Polynomial",ylab="RMSE",type = "l")#use this for report


