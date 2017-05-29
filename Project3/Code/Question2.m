file = 'ratings.data';
delim = ('\t');
a = dlmread(file, delim);
ind = crossvalind('Kfold', 100000, 10);

%To store absolute errors
error = zeros(10,1,3); 
K = [10, 50, 100];

for r=1:3
    for p=1:10
        Test = zeros(10000,5);
        TrainR = NaN(943,1682);
        TrainW = NaN(943,1682);
        k = 1;
        for q = 1:100000
            if(ind(q) ~= p)
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
        [U, V] = wnmfrule(TrainR, K(r)); 
        UV = U * V;
        for q = 1:10000
            %Predicting values
            Test(q,4) = UV(Test(q,1),Test(q,2)); 
            
            %Absolute difference between actual and predicted
            Test(q,5) = abs(Test(q,3) - Test(q,4)); 
        end
        
        x = sum(Test(:,5)); %Summing up the absolute errors for each fold
        
        x = x/10000;
        
        error(p, 1, r) = x;
    end
end


% Average error for k = 10 is 1.250763e+03
% Max error for k = 10 is 1.242835e+04
% Min error for k = 10 is 7.824958e-01


% Average error for k = 50 is 9.324846e-01
% Max error for k = 50 is 9.603764e-01
% Min error for k = 50 is 9.137594e-01


% Average error for k = 100 is 9.420888e-01
% Max error for k = 100 is 9.578691e-01
% Min error for k = 100 is 9.144101e-01

