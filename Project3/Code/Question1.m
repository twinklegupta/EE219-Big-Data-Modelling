file = 'ratings.data';
delim = ('\t');
a = dlmread(file, delim);

%Initializing matrix with zeroes
R = NaN(943, 1682);

%Not being used, but declaring it for uniformity
W = NaN(943, 1682);

%Loading matrix R using the given dataset
for p=1:size(a, 1)
    R(a(p,1), a(p,2)) = a(p,3);
    W(a(p,1), a(p,2)) = 1;
end

k = [10, 50, 100];
error = zeros(1, 3);
resid = zeros(1, 3);

for a = 1:3
    [U, V, ~, ~, resid(a)] = wnmfrule(R, k(a));
    UV = U * V;
    %Looping through all rows or users
    for p = 1:943
        %Looping through all columns or movies
        for q = 1:1682
            if isnan(R(p,q))==0
                error(a) = error(a) + (R(p, q) - UV(p, q))^2;
            end
        end
    end
    s=sprintf('The final residual is %0.4d.\n The final error is %0.4d',resid(a),error(a));
    disp(s);
end
 