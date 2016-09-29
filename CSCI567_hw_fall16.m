f_data = 'housing.data';
col_name = cellstr(char('CRIM   ','ZN     ','INDUS  ','CHAS   ','NOX    ','RM     ','AGE    ','DIS    ','RAD    ','TAX    ','PTRATIO','B      ','LSTAT  ','MEDV   '));
delimiterIn = ' ';
orig_data = importdata(f_data,delimiterIn);
% c_names = importdata(f_name,delimiterIn);

% searate train and test data
o_train = [];
o_test = [];
for i = 0:length(orig_data(:,1))-1
   if mod(i,7)==0 
       o_test = [o_test;orig_data(i+1,:)];
   else
       o_train = [o_train;orig_data(i+1,:)];
   end
end

% length(o_train) + length(o_test)
%  col_name{2}
pcor = [];
for i = 1:length(o_train(1,:))-1
    pcor = [pcor;corr(o_train(:,i),o_train(:,14))];
    f = figure;
    histogram(o_train(:,i),10);
    title(sprintf('Histogram of %s distribution\nPearson Correlation = %f', col_name{i}, pcor(i)));
    xlabel(col_name{i});
    ylabel('Count');
    s = sprintf('%s.jpg',col_name{i});
    
%     saveas(f,s);
end

% Data preprocessing for train
norm_train = o_train;
[Z,mu,sigma] = zscore(norm_train(:,1:13));

tmp = [Z,norm_train(:,14)];
norm_train  = tmp;
