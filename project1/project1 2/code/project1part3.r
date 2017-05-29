
#Piece Wise Linear Regression
#Segrgating dataset based on workflows

dataWF0 = network_backup_dataset[network_backup_dataset$Work.Flow.ID == "work_flow_0",]
dataWF1 = network_backup_dataset[network_backup_dataset$Work.Flow.ID == "work_flow_1",]
dataWF2 = network_backup_dataset[network_backup_dataset$Work.Flow.ID == "work_flow_2",]
dataWF3 = network_backup_dataset[network_backup_dataset$Work.Flow.ID == "work_flow_3",]
dataWF4 = network_backup_dataset[network_backup_dataset$Work.Flow.ID == "work_flow_4",]

#Training piecewise linear models for each workflow
linear_model_0 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=dataWF0);
linear_model_1 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=dataWF1)
linear_model_2 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=dataWF2)
linear_model_3 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=dataWF3)
linear_model_4 <- lm(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
              data=dataWF4)

#finding rmse of each model
linear_model_0_rmse <- sqrt(sum((dataWF0$Size.of.Backup..GB.-linear_model_0$fitted.values)^2)/nrow(dataWF0))

linear_model_1_rmse <- sqrt(sum((dataWF1$Size.of.Backup..GB.-linear_model_1$fitted.values)^2)/nrow(dataWF1))

linear_model_2_rmse <- sqrt(sum((dataWF2$Size.of.Backup..GB.-linear_model_2$fitted.values)^2)/nrow(dataWF2))

linear_model_3_rmse <- sqrt(sum((dataWF3$Size.of.Backup..GB.-linear_model_3$fitted.values)^2)/nrow(dataWF3))

linear_model_4_rmse <- sqrt(sum((dataWF4$Size.of.Backup..GB.-linear_model_4$fitted.values)^2)/nrow(dataWF4))
rmse <- c(linear_model_0_rmse,linear_model_1_rmse,linear_model_2_rmse,linear_model_3_rmse,linear_model_4_rmse)

mean(rmse)
plot(rmse,xlab="Work Flows",ylab="RMSE Values",type='h')

## Results : mean rmse = 0.04169366485
## random forest without wfID = 0.04180376493
## random forest with workflow id = 0.03276811397



#******************************************************************************#
#                                PolynomialRegression                          #
#******************************************************************************#


modified_dataset <- network_backup_dataset
dayList <- levels(network_backup_dataset$Day.of.Week)[c(2,6,7,5,1,3,4)]

numeric_day <- as.numeric(factor(network_backup_dataset$Day.of.Week,dayList))
modified_dataset$Day.of.Week <- numeric_day


workflowList <- levels(network_backup_dataset$Work.Flow.ID)[c(1,2,3,4,5)]
numeric_wf <- as.numeric(factor(network_backup_dataset$Work.Flow.ID,workflowList))
modified_dataset$Work.Flow.ID <- numeric_wf


fileList <- mixedsort(levels(network_backup_dataset$File.Name))
numeric_file <- as.numeric(factor(network_backup_dataset$File.Name,fileList))
modified_dataset$File.Name <- numeric_file

dataset<- modified_dataset


## fixed training set and test set, training set = 90%
trainData = dataset[1:16729,]
testData = dataset[16730:nrow(dataset),]
all_rmse = NULL

for(deg in 1:15)
{
  fitted_values = matrix(data=NA,nrow=nrow(testData),ncol=1)
  rmse = NULL
  
    
   
    pr_model <- lm(Size.of.Backup..GB. ~ polym((Day.of.Week), Backup.Start.Time...Hour.of.Day, Backup.Time..hour.,degree = deg,raw = TRUE),data=trainData)
  
    predicted_values <- predict(pr_model,testData)
    name <- rownames(as.matrix(predicted_values))
    i <- 1;
    for(j in 1:length(predicted_values))
    {
      fitted_values[i] = predicted_values[j] 
      i <- i+1
    }
    actual_values <- dataset$Size.of.Backup..GB.
    actual_values <- actual_values[16730:nrow(dataset)]
    rmse <- sqrt(mean((actual_values-fitted_values)^2))
    all_rmse <- append(all_rmse,rmse)
  }
 



plot(all_rmse,xlab="Degree of Polynomial",ylab="RMSE",type = "l")#use this for report

###################################################################################


shuffled_data<-dataset[sample(nrow(dataset)),] #Randomly distributing the data

#Create 10 equally size folds
folds <- cut(seq(1,nrow(shuffled_data)),breaks=10,labels=FALSE)


all_rmse = NULL

#10 fold cross validation
for(deg in 1:10)
{
  fitted_values = matrix(data=NA,nrow=nrow(dataset),ncol=1)
  rmse = NULL
  for(fold in 1:10){
    
    test_indices <- which(folds==fold,arr.ind=TRUE)
    trainData <- (shuffled_data[-test_indices, ])
    pr_model <- lm(Size.of.Backup..GB. ~ polym((Day.of.Week), Backup.Start.Time...Hour.of.Day, Backup.Time..hour.,degree = deg,raw = TRUE),data=trainData)
    testData <- shuffled_data[test_indices, ]
    predicted_values <- predict(pr_model,testData)
    name <- rownames(as.matrix(predicted_values))
    for(j in 1:length(predicted_values))
    {
      fitted_values[as.numeric(name[j])] = predicted_values[j] 
    }
  }
  actual_values <- dataset$Size.of.Backup..GB.
  rmse <- sqrt(mean((actual_values-fitted_values)^2))
  all_rmse <- append(all_rmse,rmse)
}
plot(all_rmse[1:8],xlab="Degree of Polynomial",ylab="RMSE")
plot(all_rmse,xlab="Degree of Polynomial",ylab="RMSE")
plot(all_rmse[1:8],xlab="Degree of Polynomial",ylab="RMSE",type="l") #use this for report
plot(all_rmse,xlab="Degree of Polynomial",ylab="RMSE",type = "l")#use this for report
