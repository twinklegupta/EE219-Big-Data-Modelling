library(randomForest)

#network_backup_dataset$numeric.workflow<-as.numeric(factor(network_backup_dataset$Work.Flow.ID, levels(network_backup_dataset$Work.Flow.ID)))
depth = 4
ns=nrow(network_backup_dataset)/depth

#Training model with depth 4 and number of tree = 20 to see the most important parameters
rf_test <- randomForest(Size.of.Backup..GB. ~ .,
                        data = network_backup_dataset,ntree=20,nodesize=ns)#Training the Model
actual_values <- network_backup_dataset$Size.of.Backup..GB.

rf_test$importance # most important parameters were Day of Week, Start time, backup time and workflow id

RMSEtoPlot = NULL
nTrees = NULL
nodeSize = NULL

for(i in 0:300)
{
  nt = 20+i
  # rf_model <- randomForest(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
  #                          data = network_backup_dataset,ntree=nt,nodesize=ns)#Training the Model
  rf_model <- randomForest(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.+Work.Flow.ID,
                           data = network_backup_dataset,ntree=nt,nodesize=ns)#Training the Model
  nTrees = append(nTrees,nt)
  nodeSize = append(nodeSize,ns)
  ns = ns - 20
  rf_predicted_values <- rf_model$predicted
  rf_rmse <- sqrt(sum((actual_values-rf_predicted_values)^2)/nrow(network_backup_dataset))
  RMSEtoPlot = append(RMSEtoPlot,rf_rmse)
}
##Plotting
plot(RMSEtoPlot, ylab="RMSE")
plot(y=RMSEtoPlot,x=nTrees, ylab="RMSE",xlab ="No of Trees")
plot(y=RMSEtoPlot,x=nodeSize, ylab="RMSE",xlab ="Node Size")



min(RMSEtoPlot, na.rm = TRUE)




## without workflow id
RMSEtoPlot = NULL
nTrees = NULL
nodeSize = NULL

for(i in 0:300)
{
  nt = 20+i
  # rf_model <- randomForest(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
  #                          data = network_backup_dataset,ntree=nt,nodesize=ns)#Training the Model
  rf_model <- randomForest(Size.of.Backup..GB. ~ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
                           data = network_backup_dataset,ntree=nt,nodesize=ns)#Training the Model
  nTrees = append(nTrees,nt)
  nodeSize = append(nodeSize,ns)
  ns = ns - 20
  rf_predicted_values <- rf_model$predicted
  rf_rmse <- sqrt(sum((actual_values-rf_predicted_values)^2)/nrow(network_backup_dataset))
  RMSEtoPlot = append(RMSEtoPlot,rf_rmse)
}


min(RMSEtoPlot, na.rm = TRUE)
