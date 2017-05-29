library(neuralnet)

actual_values <- network_backup_dataset$Size.of.Backup..GB.
# converts day of week categorical data to one hot encoding 
new_dataset <- model.matrix(~Size.of.Backup..GB.+ Day.of.Week+Backup.Start.Time...Hour.of.Day+Backup.Time..hour., data=network_backup_dataset)

#train neural network

nn <- neuralnet(Size.of.Backup..GB. ~ Day.of.WeekMonday+
                  Day.of.WeekTuesday +
                  Day.of.WeekWednesday +
                  Day.of.WeekThursday +
                  Day.of.WeekSaturday +
                  Day.of.WeekSunday
                +Backup.Start.Time...Hour.of.Day+Backup.Time..hour.,
                data = new_dataset,hidden=c(4,2), threshold=0.1)            

predicted_values <- nn$net.result[[1]]
#calculating the root mean square based on actual values and values obtained from neural network  
rmse <- sqrt(sum((actual_values-predicted_values)^2)/nrow(network_backup_dataset))
plot(nn)


## layers = 2, hidden units = 4,4, rmse = 0.03297858879
## layers = 2, hidden units = 4,2, rmse = 0.03182765621
## layers = 2, hidden units = 2,2, rmse = 0.03799274183
## layers = 1, hidden units = 10, rmse = 0.03263919916
## layers = 1, hidden units = 8, rmse =  0.03411434981
## layers = 1, hidden units = 6,  rmse = 0.03628660076
## layers = 1, hidden units = 4, rmse = 0.03923748097
## layers = 1, hidden units = 2, rmse = 0.04681281034