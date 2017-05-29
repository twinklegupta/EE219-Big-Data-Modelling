file = 'ratings.data';
delim = ('\t');
a = dlmread(file, delim);

q = size(a,1);

users = 943;
movies = 1682;

K = [10, 50, 100];
lambda = [0.01,0.1,1];

option.iter = 150;
squaredError = zeros(3);
squaredErrorSwapped = zeros(3);

R = NaN(users, movies);         
W = zeros(users, movies);       

SwappedR = R;
SwappedW = W;

% Update R and W by loading the datset
for i = 1:q
    R(a(i,1), a(i,2)) = a(i,3);
    W(a(i,1), a(i,2)) = 1;
end

% Updating R and W swapped
for i = 1:q
    SwappedR(a(i,1), a(i,2)) = a(i,3);
    if isnan(SwappedR(a(i,1), a(i,2))) == 0
        SwappedR(a(i,1), a(i,2)) = 1;
    end
    SwappedW(a(i,1), a(i,2)) = a(i,3);
end

for q = 1:3

    % For unswapped R and W
    [U,V] = wnmfrule(R,K(q));
    UV = U*V;

    % For swapped R and W
    [U_swap,V_swap] = wnmfrule(SwappedR,K(q));
    UV_swap = U_swap*V_swap;

    for i = 1:users
        for j = 1:movies
            if isnan(R(i,j))==0
                squaredError(q) = squaredError(q) + W(i,j)*(R(i,j) - UV(i,j))^2;
            end
            
            if isnan(SwappedR(i,j))==0
                squaredErrorSwapped(q) = squaredErrorSwapped(q) + SwappedW(i,j)*(SwappedR(i,j) - UV_swap(i,j))^2;
            end
        end
    end
end

% Intitializing variable for calculation of regularized error, precision
% and recall
Thresh = linspace(0.1, 5.0, 50); 

unregularizedError = zeros(3);
regularizedError = zeros(3,3);

unregularizedPrecision = zeros(50,1,3);
regularizedPrecision = zeros(50,1,3,3);

unregularizedRecall = zeros(50,1,3);
regularizedRecall = zeros(50,1,3,3);

% For regularized, calculating Precision and Recall
for p = 1:3
    for q = 1:3
        [regU, regV] = reg_wnmfrule(R, W, K(q), lambda(p), option);
        regP   = regU * regV;

        for i=1:users
            for j=1:movies
                if isnan(R(i,j))==0
                    regularizedError(q,p) = regularizedError(q,p) + W(i,j)*(R(i,j) - regP(i,j))^2;
                end
            end
        end

        % Calculate the Precision and Recall for each type of error
        for th = 1:50                    
            regularizedPrecision(th,1,q,p) = length(find((regP(:, :)>Thresh(th)) & (SwappedR==1)))/length(find(regP(:, :)>Thresh(th)));
            regularizedRecall(th,1,q,p)    = length(find((regP(:, :)>Thresh(th)) & (SwappedR==1)))/length(find(SwappedR==1));
        end
    end
end

% For unregularized, calculating Precision and Recall
for q = 1:3
    [unregU, unregV] = wnmfrule(R, K(q), option); 
    unregP = unregU * unregV;

    for i=1:users
        for j=1:movies
            if isnan(R(i,j))==0
                unregularizedError(q) = unregularizedError(q) + W(i,j)*(R(i,j) - regP(i,j))^2;
            end
        end
    end

    for th=1:50
        unregularizedPrecision(th,1,q) = length(find((unregP(:, :)>Thresh(th)) & (R>3)))/length(find(unregP(:, :)>Thresh(th)));
        unregularizedRecall(th,1,q)    = length(find((unregP(:, :)>Thresh(th)) & (R>3)))/length(find(R>3));                
    end
end

% Plotting graphs
figure;
plot(unregularizedRecall(:,1,1), unregularizedPrecision(:,1,1),'blue');
hold on;
plot(unregularizedRecall(:,1,2), unregularizedPrecision(:,1,2),'green');
hold on;
plot(unregularizedRecall(:,1,3), unregularizedPrecision(:,1,3),'red');
hold on;
title('For K = 10, 50 and 100, Precision vs Recall : Unregularised');
xlabel('Recall');
ylabel('Precision');

figure;
plot(unregularizedRecall(:,1,1), unregularizedPrecision(:,1,1),'green'); hold on;
plot(regularizedRecall(:,1,1,1), regularizedPrecision(:,1,1,1),'blue'); hold on;
plot(regularizedRecall(:,1,1,2), regularizedPrecision(:,1,1,2),'red'); hold on;
plot(regularizedRecall(:,1,1,3), regularizedPrecision(:,1,1,3),'black'); hold on;
title('For K = 10, Precision vs Recall : Regularised vs Unregularised');
xlabel('Recall');
ylabel('Precision');

figure;
plot(unregularizedRecall(:,1,2), unregularizedPrecision(:,1,2),'green'); hold on;
plot(regularizedRecall(:,1,2,1), regularizedPrecision(:,1,2,1),'blue'); hold on;
plot(regularizedRecall(:,1,2,2), regularizedPrecision(:,1,2,2),'red'); hold on;
plot(regularizedRecall(:,1,2,3), regularizedPrecision(:,1,2,3),'black'); hold on;
title('For K = 50, Precision vs Recall : Regularised vs Unregularised');
xlabel('Recall');
ylabel('Precision');

figure;
plot(unregularizedRecall(:,1,3), unregularizedPrecision(:,1,3),'green'); hold on;
plot(regularizedRecall(:,1,3,1), regularizedPrecision(:,1,3,1),'blue'); hold on;
plot(regularizedRecall(:,1,3,2), regularizedPrecision(:,1,3,2),'red'); hold on;
plot(regularizedRecall(:,1,3,3), regularizedPrecision(:,1,3,3),'black');hold on;
title('For K = 100, Precision vs Recall: Regularised vs Unregularised');
xlabel('Recall');
ylabel('Precision');
