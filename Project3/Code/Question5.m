file = 'ratings.data';
delim = ('\t');
a = dlmread(file, delim);

users = 943;
movies = 1682;
K = [10, 50, 100];

crossvalidindices = crossvalind('Kfold', 100000, 10);
predictedValues = NaN(users, movies, 3); 

% Matrixes for storing Precision, Hit Rate and False Alarm Rat
precision = zeros(users,3); 
hitRate = zeros(users,25,3); 
falseAlarmRate = zeros(users,25,3);

for n=1:3
    for p = 1:10
        test = zeros(10000,5);
        TrainR = NaN(users, movies);
        TrainW = NaN(users, movies);
        t = 1;
        for q = 1:100000
            
            if(crossvalidindices(q) ~= p)
                TrainW(a(q,1), a(q,2)) = a(q,3); 
                TrainR(a(q,1), a(q,2)) = 1;
            
            % Storing the Test Data in 2D matrix
            else
                test(t,1) = a(q,1); 
                test(t,2) = a(q,2); 
                test(t,3) = a(q,3);
                t=t+1;
            end
        end
        
        % Matrix factorization
        [U,V] = wnmfrule(TrainR, K(n)); 
        UV = U * V;

        for q = 1:10000
            test(q,4) = UV(test(q,1),test(q,2)); 
            % Absolute difference between actual and predicted values
            test(q,5) = abs(test(q,3) - test(q,4)); 
            
            predictedValues(test(q,1),test(q,2),n) = UV(test(q,1),test(q,2));
        end
    end
end

% Values of L range from 1 to 25 and for each L Hit-Rate and False-Alarm
% Rate are found out

% Best Threshold from Question 4
thresh = 0.4;  
for n = 1:3
    for p=1:users 
    % Sorting in decreasing order
    [~, indices] = sort(predictedValues(p,:,n), 'descend');  
         for L=1:25
            top_Lmovies = zeros(users, L, 3);
            count = 1; 
            
            % If the movie is rated by user
            for t=1:size(indices,2)
                if TrainR(p,indices(t))== 1
                    top_Lmovies(p,count,n) = indices(t);
                    count = count+1;
                end
                
                % When top L movies have been found out
                if count == L+1
                    tp = length(find((predictedValues(p,top_Lmovies(p,:,n),n)> thresh) & (TrainW(p,top_Lmovies(p,:,n))>3)));
                    tn = length(find((predictedValues(p,top_Lmovies(p,:,n),n)<= thresh) & (TrainW(p,top_Lmovies(p,:,n))<=3)));
                    fp = length(find((predictedValues(p,top_Lmovies(p,:,n),n)> thresh) & (TrainW(p,top_Lmovies(p,:,n))<=3)));
                    fn = length(find((predictedValues(p,top_Lmovies(p,:,n),n)<= thresh) & (TrainW(p,top_Lmovies(p,:,n))>3)));
                   
                    % Special case for L = 5
                    if L == 5
                        precision(p,n) = tp/length(find(predictedValues(p, top_Lmovies(p,:,n),n)> thresh));
                    end
                    
                    % Hit rate
                    if tp==0 && fn==0
                       hitRate(p,L,n) = 0; 
                    else
                        hitRate(p,L,n) = tp/(tp+fn);
                    end
                    
                    % False-alarm rate
                    if fp==0 && tn==0
                        falseAlarmRate(p,L,n) = 0; 
                    else
                        falseAlarmRate(p,L,n) = fp/(fp+tn);
                    end
                    
                    break;    
                end
            end
        end
    end 
end

% Average precision for L = 5
avgPrecision = zeros(3);
for n=1:3
    avgPrecision(n) =  mean(precision(:,n));
end

msg = sprintf('\nFor L = 5:');
disp(msg);
for n = 1:3
    msg = sprintf('K = %d\t Avg. precision = %d', K(n), avgPrecision(n));
    disp(msg);
end

% Mean hit rate and false-alarm rate
meanHitRate = zeros(25,3); 
meanFalseAlarmRate = zeros(25,3);
for n = 1 : 3
    for L = 1 : 25
        meanHitRate(L,n) = mean(hitRate(:,L,n));
        meanFalseAlarmRate(L,n) = mean(falseAlarmRate(:,L,n));
    end
end

for n = 1 : 3
    msg = sprintf('\n\nK = %d\nL\tMean Hit Rate\tMean False-alarm Rate', K(n));
    disp(msg);
    for p = 1 : 25
        msg = sprintf('%d\t%d\t%d', p, meanHitRate(p,1), meanFalseAlarmRate(p,1));
        disp(msg);
    end
end

% Graph Plots

figure;
plot((1:25),meanFalseAlarmRate(:,1),'blue'); hold on;
plot((1:25),meanFalseAlarmRate(:,2),'green'); hold on;
plot((1:25),meanFalseAlarmRate(:,3),'red'); hold on;
title('Average False Alarm Rate vs L');
xlabel('L')
ylabel('Average False Alarm Rate')

figure;
plot((1:25),meanHitRate(:,1),'blue'); hold on;
plot((1:25),meanHitRate(:,2),'green'); hold on;
plot((1:25),meanHitRate(:,3),'red'); hold on;
title('Average Hit Rate vs L');
xlabel('L')
ylabel('Average Hit Rate')

figure;
plot(meanFalseAlarmRate(:,1),meanHitRate(:,1),'blue'); hold on;
plot(meanFalseAlarmRate(:,2),meanHitRate(:,2),'green'); hold on;
plot(meanFalseAlarmRate(:,3),meanHitRate(:,3),'red'); hold on;
title('Average Hit Rate vs Average False Alarm Rate');
xlabel('Average False Alarm Rate')
ylabel('Average Hit Rate')