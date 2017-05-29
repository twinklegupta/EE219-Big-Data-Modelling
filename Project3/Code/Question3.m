file = 'ratings.data';
delim = ('\t');
a = dlmread(file, delim);

R = NaN(943, 1682);

n = size(a, 1);

K = [10, 50, 100];

%Loading R from the dataset
for p=1:n
    R(a(p,1), a(p,2)) = a(p,3);
end

ind = crossvalind('Kfold', 100000, 10);

%Predicted values
PredictR = NaN(943,1682,3); 

%Defining threshold vector
Thresh = linspace(0.1, 5.0, 50);

Precision = zeros(50, 1, 3);
Recall = zeros(50, 1, 3);

for r = 1:3
    for p = 1:10
        Test = zeros(10000,3); %Will store 10% of the data
        TrainR = NaN(943,1682); %Will store 90% of the data
        TrainW = NaN(943,1682);
        k = 1;
        
        for q = 1:100000
            if(p ~= ind(q))
                %Creating the training matrix for R
                TrainR(a(q,1),a(q,2)) = a(q,3);
                TrainW(a(q,1),a(q,2)) = 1;
            else
                %Test matrix
                Test(k,1) = a(q,1);	
                Test(k,2) = a(q,2); 
                Test(k,3) = a(q,3); 
                
                k=k+1;
            end
        end
        
        [U, V] = wnmfrule(TrainR,K(r));
        UV = U * V;
        
        for q = 1:10000
            
            %Storing precited value
            PredictR(Test(q,1),Test(q,2),r) = UV(Test(q,1),Test(q,2)); 
        end
    end
end

%Precision and Recall for threshold
for r = 1:3
    for p = 1:50
        Precision(p, 1, r) = length(find((PredictR(:, :,r) > Thresh(p)) & (R>3)))/length(find(PredictR(:, :,r)>Thresh(p)));
        Recall(p, 1, r) = length(find((PredictR(:, :,r) > Thresh(p)) & (R>3)))/length(find(R>3));
    end
end

%Plotting results
figure;
plot(Recall(:,1,1), Precision(:,1,1),'Marker','o','MarkerFaceColor','green')
title('K=10: Precision v/s Recall')
xlabel('Recall')
ylabel('Precision')

figure;
plot(Recall(:,1,2), Precision(:,1,2),'Marker','o','MarkerFaceColor','green')
title('K=50: Precision v/s Recall')
xlabel('Recall')
ylabel('Precision')

figure;
plot(Recall(:,1,3), Precision(:,1,3),'Marker','o','MarkerFaceColor','green')
title('K=100: Precision v/s Recall ')
xlabel('Recall')
ylabel('Precision')

figure;
plot(Thresh(:), Precision(:,1,1),'Marker','o','MarkerFaceColor','blue')
title('K=10: Precision v/s Threshold')
xlabel('Threshold')
ylabel('Precision')

figure;
plot(Thresh(:), Precision(:,1,2),'Marker','o','MarkerFaceColor','blue')
title('K=50: Precision v/s Threshold')
xlabel('Threshold')
ylabel('Precision')

figure;
plot(Thresh(:), Precision(:,1,3),'Marker','o','MarkerFaceColor','blue')
title('k=100: Precision v/s Threshold')
xlabel('Threshold')
ylabel('Precision')


figure;
plot(Thresh(:), Recall(:,1,1),'Marker','o','MarkerFaceColor','red')
title('K=10: Recall v/s Threshold')
xlabel('Threshold')
ylabel('Recall')

figure;
plot(Thresh(:), Recall(:,1,2),'Marker','o','MarkerFaceColor','red')
title('K=50: Recall v/s Threshold')
xlabel('Threshold')
ylabel('Recall')

figure;
plot(Thresh(:), Recall(:,1,3),'Marker','o','MarkerFaceColor','red')
title('K=100: Recall v/s Threshold')
xlabel('Threshold')
ylabel('Recall')