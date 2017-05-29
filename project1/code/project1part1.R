
#converting days of the week into integer values
dayList <- levels(network_backup_dataset$Day.of.Week)[c(2,6,7,5,1,3,4)]

numeric_day <- as.numeric(factor(network_backup_dataset$Day.of.Week,dayList))

#appending the numberic day column to original dataset
network_backup_dataset$numeric.day <- numeric_day

#calculating the day number depending on the week number
day_number = ((network_backup_dataset$Week.. -1)*7)+network_backup_dataset$numeric.day

#appending the day number to original dataset
network_backup_dataset$day.number <- day_number


#create a new dataset only for 20 days
new_dataset = network_backup_dataset[with(network_backup_dataset,day.number >= 21 & day.number <=40), ]

#sort the new dataset with respect to workflow id
new_dataset = new_dataset[order(new_dataset$Work.Flow.ID),]
#segregate data according to workglow
workflow0_filesize = new_dataset[with(new_dataset,Work.Flow.ID == 'work_flow_0'), ]
workflow1_filesize = new_dataset[with(new_dataset,Work.Flow.ID == 'work_flow_1'), ]
workflow2_filesize = new_dataset[with(new_dataset,Work.Flow.ID == 'work_flow_2'), ]
workflow3_filesize = new_dataset[with(new_dataset,Work.Flow.ID == 'work_flow_3'), ]
workflow4_filesize = new_dataset[with(new_dataset,Work.Flow.ID == 'work_flow_4'), ]


dataframe0 <- data.frame(workflow0_filesize)
DT <- data.table(workflow0_filesize)
sum1 <- DT[,sum(Size.of.Backup..GB.), by = day.number]
dataframe0 <- data.frame(workflow1_filesize)
DT <- data.table(workflow1_filesize)
sum2 <- DT[,sum(Size.of.Backup..GB.), by = day.number]
dataframe0 <- data.frame(workflow2_filesize)
DT <- data.table(workflow2_filesize)
sum3 <- DT[,sum(Size.of.Backup..GB.), by = day.number]
dataframe0 <- data.frame(workflow3_filesize)
DT <- data.table(workflow3_filesize)
sum4 <- DT[,sum(Size.of.Backup..GB.), by = day.number]
dataframe0 <- data.frame(workflow4_filesize)
DT <- data.table(workflow4_filesize)
sum5 <- DT[,sum(Size.of.Backup..GB.), by = day.number]

#plot the graph

ggplot(sum1, aes(x = day.number))+ geom_line(aes(y = V1),color="blue") +
            labs(x="Number of Days", y="Size of Backup (GB)") + 
              scale_x_continuous(breaks = seq(1,20,1)) + ggtitle("Workflow 0")
ggplot(sum2, aes(x = day.number))+ geom_line(aes(y = V1),color="blue") +
            labs(x="Number of Days", y="Size of Backup (GB)") +
            scale_x_continuous(breaks = seq(1,20,1))  + ggtitle("Workflow 1")
ggplot(sum3, aes(x = day.number))+ geom_line(aes(y = V1),color="blue") +
            labs(x="Number of Days", y="Size of Backup (GB)") +
            scale_x_continuous(breaks = seq(1,20,1))  + ggtitle("Workflow 2")
ggplot(sum4, aes(x = day.number))+ geom_line(aes(y = V1),color="blue") +
            labs(x="Number of Days", y="Size of Backup (GB)")   +
            scale_x_continuous(breaks = seq(1,20,1))  + ggtitle("Workflow 3")
  
ggplot(sum5, aes(x = day.number))+ geom_line(aes(y = V1),color="blue") +
            labs(x="Number of Days", y="Size of Backup (GB)") +
            scale_x_continuous(breaks = seq(1,20,1))  + ggtitle("Workflow 4")