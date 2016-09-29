f_data = 'housing.data';
col_name = cellstr(char('CRIM   ','ZN     ','INDUS  ','CHAS   ','NOX    ','RM     ','AGE    ','DIS    ','RAD    ','TAX    ','PTRATIO','B      ','LSTAT  ','MEDV   '));
delimiterIn = ' ';
orig_data = importdata(f_data,delimiterIn);
L = [0.01, 0.1, 1.0];
str_res = cellstr(char('Linear Regression        ','Rigde Regression  L =0.01','Rigde Regression  L =0.10','Rigde Regression  L =1.00'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% separate train and test data
o_train = [];
o_test = [];
for i = 0:length(orig_data(:,1))-1
    if mod(i,7)==0
        o_test = [o_test;orig_data(i+1,:)];
    else
        o_train = [o_train;orig_data(i+1,:)];
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % length(o_train) + length(o_test)
% %  col_name{2}

%PLOT HISTOGRAMS AND CALCULATE PERSON CORELLATION

%UNCOMMENT BELOW FOR LOOP LATER
% pcor = [];
% for i = 1:length(o_train(1,:))-1
%     pcor = [pcor;corr(o_train(:,i),o_train(:,14))];
%     f = figure;
%     histogram(o_train(:,i),10);
%     title(sprintf('Histogram of %s distribution\nPearson Correlation = %f', col_name{i}, pcor(i)));
%     xlabel(col_name{i});
%     ylabel('Count');
%     s = sprintf('%s.jpg',col_name{i});
%
% %     saveas(f,s);
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Data preprocessing for train
Res_train = cell(4,2);
X_norm_train = o_train(:,1:13);
Y_train_true = o_train(:,14);
[Z_train,mu_train,sigma_train] = zscore(X_norm_train);
X_norm_train = Z_train;
sz_train = size(X_norm_train(:,1));
o_1_train = ones(sz_train);
X_norm_train = [o_1_train,X_norm_train];


%Linear Regression for TRAIN
W_train_lr = [];
[W_train_lr, Y_pred_train] = Linear_regression(X_norm_train, Y_train_true);

MSE_lr_train = ( sum( (Y_train_true - Y_pred_train).^2) ) / length(X_norm_train(:,1));
Res_train(1,1) = str_res(1);
Res_train(1,2) = cellstr(num2str(MSE_lr_train));

% fprintf('\nThe MSE for linear regression for training set is :: %f\n\n',MSE_lr_train);

%Ridge Regression for TRAIN
W_train_rr = [];
for i =1:length(L)
    
    [W_rr, Y_pred_train_rr] = Ridge_regression(X_norm_train, L(i), Y_train_true);
    W_train_rr = [W_train_rr; W_rr];
    MSE_rr_train = ( sum( (Y_train_true - Y_pred_train_rr).^2) ) / length(X_norm_train(:,1));
    Res_train(i+1,1) = str_res(i+1);
    Res_train(i+1,2) = cellstr(num2str(MSE_rr_train));
    
    %     fprintf('\nThe MSE for ridge regression for Lamda = %f on training set is :: %f\n\n',L(i),MSE_rr_train);
end

fprintf('_____________________________________________________________________\n');
fprintf('The results for Training Set using Linear and Ridge regression::\n');
fprintf('\t\tAlgorithm\t\t\t\t\tMSE\n')
disp(Res_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Preprocessing For Training
Res_test = cell(4,2);
X_norm_test = o_test(:,1:13);
Y_test_true = o_test(:,14);

for i = 1: length(X_norm_test(:,1))
    X_norm_test(i,:) = ((X_norm_test(i,:) - mu_train))./(sigma_train);
end

sz_test = size(X_norm_test(:,1));
o_1_test = ones(sz_test);
X_norm_test = [o_1_test,X_norm_test];


%Linear Regression for TEST
Y_lr_test = [];
for i =1:length(X_norm_test(:,1))
    temp_y = W_train_lr * transpose(X_norm_test(i,:));
    Y_lr_test = [Y_lr_test;temp_y];
end
% [W_test, Y_pred_test] = Linear_regression(X_norm_test, Y_test_true);

MSE_lr_test = ( sum( (Y_test_true - Y_lr_test).^2) ) / length(X_norm_test(:,1));
Res_test(1,1) = str_res(1);
Res_test(1,2) = cellstr(num2str(MSE_lr_test));

% fprintf('\nThe MSE for linear regression for testing set is :: %f\n\n',MSE_lr_test);


%Ridge Regression for TEST

for i =1:length(L)
    Y_rr_test = [];
%     [W_train_rr, Y_pred_test_rr] = Ridge_regression(X_norm_test, L(i), Y_test_true);
    
    W_rr_l = W_train_rr(i,:);
    for j =1:length(X_norm_test(:,1))
        temp_y = W_rr_l * transpose(X_norm_test(j,:));
        Y_rr_test = [Y_rr_test;temp_y];
    end
    
    MSE_rr_test = ( sum( (Y_test_true - Y_rr_test).^2) ) / length(X_norm_test(:,1));
    Res_test(i+1,1) = str_res(i+1);
    Res_test(i+1,2) = cellstr(num2str(MSE_rr_test));
    
    %fprintf('\nThe MSE for ridge regression for Lamda = %f on testing set is :: %f\n\n',L(i),MSE_rr_test);
end

fprintf('\n\n_____________________________________________________________________\n');
fprintf('The results for Testing Set using Linear and Ridge regression::\n');
fprintf('\t\tAlgorithm\t\t\t\t\tMSE\n')
disp(Res_test);
fprintf('_____________________________________________________________________\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ridge Regression with Cross-Validation
%10 fold cross validation
%(43)x7 + (44)x3

% Train_cell = mat2cell([X_norm_train,Y_train_true],[43,43,43,43,43,43,43,44,44,44],[14,1]);
%
% Train_cell{1,1}
% Train_cell{1,2}

% index = 1
% for i=1:10
%         if(i<=7)
%            cv_X_test = X_norm_train(index:index+42,:);
%            index = index + 43
%
%         else
%             cv_X_test = X_norm_train(index:index+43,:);
%             index = index + 44
%         end
% end







% 
% 
% cv_lamda = 0.0001;
% Res_cv = [];
% while(abs(cv_lamda - 10) > 0.00001)
%     %retain 43 rows 7 times and 44 rows 3 times
%     cv_X_norm_train = X_norm_train;
%     cv_Y_train_true = Y_train_true;
%     index = 1;
%     for i=1:10
%         if(i<=7)
%             %extract the test data
%             cv_X_test = X_norm_train(index:index+42,:);
%             cv_Y_test_true = Y_train_true(index:index+42);
%             %delete the extracted data from training set
%             cv_X_norm_train(index:index+42,:) = [];
%             cv_Y_train_true(index:index+42) = [];
%             index = index + 43;
%         else
%             cv_X_test = X_norm_train(index:index+43,:);
%             cv_Y_test_true = Y_train_true(index:index+43);
%             cv_X_norm_train(index:index+43,:) = [];
%             cv_Y_train_true(index:index+43) = [];
%             index = index + 44;
%         end
%         
%         [W_test_rr, Y_pred_test_rr] = Ridge_regression(cv_X_norm_train, cv_lamda, cv_Y_train_true);
%         MSE_rr_test = ( sum( (cv_Y_train_true - Y_pred_test_rr).^2) ) / length(cv_X_norm_train(:,1));
%         Res_cv = [Res_cv; cv_lamda,MSE_rr_test];
%         
%     end
%     cv_lamda = cv_lamda *2;
% end
