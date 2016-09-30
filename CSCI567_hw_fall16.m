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

% % % length(o_train) + length(o_test)
% % %  col_name{2}

%PLOT HISTOGRAMS AND CALCULATE PERSON CORELLATION

%UNCOMMENT BELOW FOR LOOP LATER
pcor = [];
for i = 1:length(o_train(1,:))-1
    pcor = [pcor;i,corr(o_train(:,i),o_train(:,14))];
    
    %UNCOMMENT BELOW FOR LOOP LATER
    %     f = figure;
    %     histogram(o_train(:,i),10);
    %     title(sprintf('Histogram of %s distribution\nPearson Correlation = %f', col_name{i}, pcor(i)));
    %     xlabel(col_name{i});
    %     ylabel('Count');
    %     s = sprintf('%s.jpg',col_name{i});
    
    %     saveas(f,s);
end


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

%Ridge Regression for TRAIN
W_train_rr = [];
for i =1:length(L)
    
    [W_rr, Y_pred_train_rr] = Ridge_regression(X_norm_train, L(i), Y_train_true);
    W_train_rr = [W_train_rr; W_rr];
    MSE_rr_train = ( sum( (Y_train_true - Y_pred_train_rr).^2) ) / length(X_norm_train(:,1));
    Res_train(i+1,1) = str_res(i+1);
    Res_train(i+1,2) = cellstr(num2str(MSE_rr_train));
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

%Normalizing Test Data
for i = 1: length(X_norm_test(:,1))
    X_norm_test(i,:) = ((X_norm_test(i,:) - mu_train))./(sigma_train);
end

sz_test = size(X_norm_test(:,1));
o_1_test = ones(sz_test);
X_norm_test = [o_1_test,X_norm_test];


%Linear Regression for TEST based in W learned in training
Y_lr_test = [];
for i =1:length(X_norm_test(:,1))
    temp_y = W_train_lr * transpose(X_norm_test(i,:));
    Y_lr_test = [Y_lr_test;temp_y];
end

MSE_lr_test = ( sum( (Y_test_true - Y_lr_test).^2) ) / length(X_norm_test(:,1));
Res_test(1,1) = str_res(1);
Res_test(1,2) = cellstr(num2str(MSE_lr_test));


%Ridge Regression for TEST

for i =1:length(L)
    Y_rr_test = [];
    
    W_rr_l = W_train_rr(i,:);
    
    for j =1:length(X_norm_test(:,1))
        temp_y = W_rr_l * transpose(X_norm_test(j,:));
        Y_rr_test = [Y_rr_test;temp_y];
    end
    
    MSE_rr_test = ( sum( (Y_test_true - Y_rr_test).^2) ) / length(X_norm_test(:,1));
    Res_test(i+1,1) = str_res(i+1);
    Res_test(i+1,2) = cellstr(num2str(MSE_rr_test));
end

fprintf('\n_____________________________________________________________________\n');
fprintf('The results for Testing Set using Linear and Ridge regression::\n');
fprintf('\t\tAlgorithm\t\t\t\t\tMSE\n')
disp(Res_test);
fprintf('_____________________________________________________________________\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ridge Regression with Cross-Validation
%10 fold cross validation
fprintf('\n Results of Cross validation on Training Set::\n Lamda Value \t MSE\n');
cv_lamda = 0.0001;
Res_cv = [];
Res_cv_w = [];
res_i =1;
while(abs(cv_lamda - 10) > 0.0001 && cv_lamda <= 10)
    %retain 43 rows 7 times and 44 rows 3 times
    curr_mse_l = 0;
    index = 1;
    
    for i=1:10
        min_mse = Inf;
        w_min = [];
        cv_X_norm_train = X_norm_train;
        cv_Y_train_true = Y_train_true;
        if(i<=7)
            %extract the test data
            cv_X_test = X_norm_train(index:index+42,:);
            cv_Y_test_true = Y_train_true(index:index+42);
            %delete the extracted data from training set
            cv_X_norm_train(index:index+42,:) = [];
            cv_Y_train_true(index:index+42) = [];
            index = index + 43;
        else
            cv_X_test = X_norm_train(index:index+43,:);
            cv_Y_test_true = Y_train_true(index:index+43);
            cv_X_norm_train(index:index+43,:) = [];
            cv_Y_train_true(index:index+43) = [];
            index = index + 44;
        end
        
        [W_rr, Y_pred_test_rr] = Ridge_regression(cv_X_norm_train, cv_lamda, cv_Y_train_true);
        Y_rr_test_cv = [];
        for k =1:length(cv_X_test(:,1))
            temp_y = W_rr * transpose(cv_X_test(k,:));
            Y_rr_test_cv = [Y_rr_test_cv;temp_y];
        end
        
        MSE_rr_test = ( sum( (cv_Y_test_true - Y_rr_test_cv).^2) ) / length(cv_X_test(:,1));
        if MSE_rr_test < min_mse
            min_mse = MSE_rr_test;
            w_min = W_rr;
        end
        curr_mse_l = curr_mse_l + MSE_rr_test;
        %         Res_cv = [Res_cv; cv_lamda,MSE_rr_test];
        
    end
    Res_cv = [Res_cv; cv_lamda,(curr_mse_l/10)];
    Res_cv_w = [Res_cv_w; cv_lamda,  w_min];
    fprintf('%f\t\t%f\n',Res_cv(res_i,1),Res_cv(res_i,2));
    res_i = res_i +1;
    
    cv_lamda = cv_lamda * 10;
end

% fprintf('\n Results of Cross validation::\n Lamda Value \t MSE\n');
% disp(Res_cv);

fprintf('_____________________________________________________________________\n');
fprintf('\n Results of Cross validation on Testing Set::\n Lamda Value \t MSE\n');
Res_cv_test = [];
for i=1:length(Res_cv_w(:,1))
    curr_w = Res_cv_w(i,2:15);
    curr_l =  Res_cv_w(i,1);
    Y_rr_test = [];
    for j =1:length(X_norm_test(:,1))
        temp_y = curr_w * transpose(X_norm_test(j,:));
        Y_rr_test = [Y_rr_test;temp_y];
    end
    
    MSE_rr_test = ( sum( (Y_test_true - Y_rr_test).^2) ) / length(X_norm_test(:,1));
    Res_cv_test = [Res_cv_test; curr_l,MSE_rr_test];
    fprintf('%f\t\t%f\n',Res_cv_test(i,1),Res_cv_test(i,2));
    
    
end


%FEATURE SELECTION
%BEST 4
A = [pcor(:,1),abs(pcor(:,2))];
sorted_pcor = sortrows(A,-2);
index_p = int16(sorted_pcor(1:4,1));
index_p = index_p +1;

X_fc_train = [X_norm_train(:,1),X_norm_train(:,index_p(1)),X_norm_train(:,index_p(2)),X_norm_train(:,index_p(3)),X_norm_train(:,index_p(4))];
Y_fc_true = Y_train_true;

[W_fc_train_lr, Y_pred_train] = Linear_regression(X_fc_train, Y_fc_true);
MSE_fc_lr_train = ( sum( (Y_fc_true - Y_pred_train).^2) ) / length(X_fc_train(:,1));
fprintf('_____________________________________________________________________\n');
fprintf('\n\nFeatures Selected based on highest correlation are::\nAttrubute\t\tCorellation\n');
disp(sorted_pcor(1:4,:));
fprintf('\nMSE on training data:: %f',MSE_fc_lr_train);

X_fc_test = [X_norm_test(:,1),X_norm_test(:,index_p(1)),X_norm_test(:,index_p(2)),X_norm_test(:,index_p(3)),X_norm_test(:,index_p(4))];


Y_fc_lr_test = [];
for i =1:length(X_fc_test(:,1))
    temp_y = W_fc_train_lr * transpose(X_fc_test(i,:));
    Y_fc_lr_test = [Y_fc_lr_test;temp_y];
end

MSE_fc_lr_test = ( sum( (Y_test_true - Y_fc_lr_test).^2) ) / length(X_fc_test(:,1));
fprintf('\nMSE on Testing data:: %f\n',MSE_fc_lr_test);

%RESIDUAL

%BRUTE FORCE
fprintf('_____________________________________________________________________\n');
fprintf('Best Columns section using Brute Force');

V = [2,3,4,5,6,7,8,9,10,11,12,13,14];
C = combnk(V,4);

bf_min_mse_test = inf;
bf_min_mse_train = inf;
bf_min_cols_test = [];
bf_min_cols_train = [];
for i=1:length(C(:,1))
    curr_cols = C(i,:);
    X_bf_train = [X_norm_train(:,1),X_norm_train(:,curr_cols(1)),X_norm_train(:,curr_cols(2)),X_norm_train(:,curr_cols(3)),X_norm_train(:,curr_cols(4))];
    Y_bf_true = Y_train_true;
    
    [W_bf_train_lr, Y_pred_train] = Linear_regression(X_bf_train, Y_bf_true);
    MSE_bf_lr_train = ( sum( (Y_bf_true - Y_pred_train).^2) ) / length(X_bf_train(:,1));
    
    if  MSE_bf_lr_train < bf_min_mse_train
        bf_min_mse_train = MSE_bf_lr_train;
        bf_min_cols_train = curr_cols;
    end
    
    
    X_bf_test = [X_norm_test(:,1),X_norm_test(:,curr_cols(1)),X_norm_test(:,curr_cols(2)),X_norm_test(:,curr_cols(3)),X_norm_test(:,curr_cols(4))];
    Y_bf_lr_test = [];
    for i =1:length(X_bf_test(:,1))
        temp_y = W_bf_train_lr * transpose(X_bf_test(i,:));
        Y_bf_lr_test = [Y_bf_lr_test;temp_y];
    end
    MSE_bf_lr_test = ( sum( (Y_test_true - Y_bf_lr_test).^2) ) / length(X_bf_test(:,1));
    if MSE_bf_lr_test < bf_min_mse_test 
        bf_min_mse_test = MSE_bf_lr_test;
        bf_min_cols_test = curr_cols;
    end
    
end
fprintf('\nBest Columns for for MIN MSE: %f on Training SET\n',bf_min_mse_train);
disp(bf_min_cols_train-1);

fprintf('\nBest Columns for for MIN MSE: %f on Training SET\n',bf_min_mse_test);
disp(bf_min_cols_test-1);

%BRUTE FORCE
fprintf('_____________________________________________________________________\n');
fprintf('Polynomial Feature Expansion');

X_pf_test = X_norm_test;
X_pf_train = X_norm_train;

for i=2: length(X_norm_test(1,:))
    for j=i: length(X_norm_test(1,:))
        new_X_train = X_norm_train(:,i) .* X_norm_train(:,j);
        X_pf_train = [X_pf_train,new_X_train];
        new_X_test = X_norm_test(:,i) .* X_norm_test(:,j);
        X_pf_test = [X_pf_test,new_X_test];
        
        
    end
end

[W_train_lr, Y_pred_train] = Linear_regression(X_pf_train, Y_train_true);

MSE_lr_train = ( sum( (Y_train_true - Y_pred_train).^2) ) / length(X_pf_train(:,1));
fprintf('\nMSE on Training data:: %f\n',MSE_lr_train);

Y_lr_test = [];
for i =1:length(X_pf_test(:,1))
    temp_y = W_train_lr * transpose(X_pf_test(i,:));
    Y_lr_test = [Y_lr_test;temp_y];
end

MSE_lr_test = ( sum( (Y_test_true - Y_lr_test).^2) ) / length(X_pf_test(:,1));
fprintf('\nMSE on Testing data:: %f\n',MSE_lr_test);

% % %  disp(Res_cv_test);
% [B, FitInfo] = lasso(X_norm_train,Y_train_true,'CV',10);
% lassoPlot(B,FitInfo,'PlotType','CV');


%(43)x7 + (44)x3

% % % Train_cell = mat2cell([X_norm_train,Y_train_true],[43,43,43,43,43,43,43,44,44,44],[14,1]);
% % %
% % % Train_cell{1,1}
% % % Train_cell{1,2}

% % % index = 1
% % % for i=1:10
% % %         if(i<=7)
% % %            cv_X_test = X_norm_train(index:index+42,:);
% % %            index = index + 43
% % %
% % %         else
% % %             cv_X_test = X_norm_train(index:index+43,:);
% % %             index = index + 44
% % %         end
% % % end



