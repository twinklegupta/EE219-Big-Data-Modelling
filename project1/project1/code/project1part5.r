shuffled_data<-housing_data[sample(nrow(housing_data)),] 
x <- shuffled_data[,1:ncol(housing_data)-1]
y <- shuffled_data[,ncol(housing_data)]

x <- as.matrix(x)
y <- as.matrix(y)



#Create 10 equally size folds
folds <- cut(seq(1,nrow(shuffled_data)),breaks=10,labels=FALSE)

####################################################################################
#                                       Ridge Regression                           #
####################################################################################
rmse_toplot = NULL


#10 fold cross validatio

lambdas = c(1.0,0.1,0.01,0.001,0.0001,0.00001)
for(l in 1:length(lambdas)) {  
  rmse = NULL
  fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
for(fold in 1:10){
    
    test_indices <- which(folds==fold,arr.ind=TRUE)
    trainData <- (x[-test_indices, ])
    trainValues <- (y[-test_indices, ])
    testData <- x[test_indices, ]
    testValues <- y[test_indices]
    #alpha = 0 for ridge regresion
    ridge_model<- glmnet(as.matrix(trainData),as.matrix(trainValues),family = "gaussian",lambda = lambdas[l], alpha = 0)
 
    predicted_values <- predict(ridge_model,testData)
    name <- rownames(as.matrix(predicted_values))
    for(j in 1:length(predicted_values))
    {
      fitted_values[as.numeric(name[j])] = predicted_values[j] 
    }
  }
  actual_values <- housing_data$MEDV
  rmse <- sqrt(sum((actual_values-fitted_values)^2)/nrow(housing_data))
  rmse_toplot = append(rmse_toplot,rmse)
}

plot(lambdas[2:6],rmse_toplot[2:6],type = "l")
  
  ## for report expalin alpha and its formula
  # rmse = 4.921144 4.861304 4.862714 4.863097 4.863142 4.8631462
  
  
  
  ####################################################################################
  #                                       Lasso Regression                           #
  ####################################################################################
  
rmse_toplot = NULL
  #10 fold cross validatio
for(l in 1 : length(lambdas))  {
  rmse = NULL
  fitted_values = matrix(data=NA,nrow=nrow(housing_data),ncol=1)
  for(fold in 1:10){
    
    test_indices <- which(folds==fold,arr.ind=TRUE)
    trainData <- (x[-test_indices, ])
    trainValues <- (y[-test_indices, ])
    testData <- x[test_indices, ]
    testValues <- y[test_indices]
    #alpha = 0 for ridge regresion
    ridge_model<- glmnet(as.matrix(trainData),as.matrix(trainValues),family = "gaussian",lambda = lambdas[l], alpha = 1.0)
    
    predicted_values <- predict(ridge_model,testData)
    name <- rownames(as.matrix(predicted_values))
    for(j in 1:length(predicted_values))
    {
      fitted_values[as.numeric(name[j])] = predicted_values[j] 
    }
  }
  actual_values <- housing_data$MEDV
  rmse <- sqrt(sum((actual_values-fitted_values)^2)/nrow(housing_data))
  rmse_toplot = append(rmse_toplot,rmse)
}
  
  
plot(lambdas[2:6],rmse_toplot[2:6],type = "l")
  
  ## for report expalin alpha and its formula
  #rmse  5.427831 4.905366 4.861268 4.862909 4.863154 4.863144




