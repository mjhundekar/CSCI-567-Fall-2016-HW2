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

%PLOT HISTOGRAMS AND CALCULATE PERSON CORELLATION

%UNCOMMENT BELOW FOR LOOP LATER
pcor = [];
for i = 1:length(o_train(1,:))-1
    pcor = [pcor;i,corr(o_train(:,i),o_train(:,14))];
    
%     UNCOMMENT BELOW FOR LOOP LATER
        f =  figure('Name',col_name{i});
        histogram(o_train(:,i),10);
        title(sprintf('Histogram of %s distribution\nPearson Correlation = %f', col_name{i}, pcor(i,2)));
        xlabel(col_name{i});
        ylabel('Count');
        s = sprintf('%s.jpg',col_name{i});
    
    %     saveas(f,s);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Data preprocessing for train

Res_train = cell(4,2);
%split attribute and Y value
X_norm_train = o_train(:,1:13);
Y_train_true = o_train(:,14);

%Normalize
[Z_train,mu_train,sigma_train] = zscore(X_norm_train);
X_norm_train = Z_train;

%Add a colum of 1's 
sz_train = size(X_norm_train(:,1));
o_1_train = ones(sz_train);
X_norm_train = [o_1_train,X_norm_train];

%Preprocessing For Testing
Res_test = cell(4,2);
X_norm_test = o_test(:,1:13);
Y_test_true = o_test(:,14);

%Normalizing Test Data
for i = 1: length(X_norm_test(:,1))
    X_norm_test(i,:) = ((X_norm_test(i,:) - mu_train))./(sigma_train);
end

%Add column of 1's
sz_test = size(X_norm_test(:,1));
o_1_test = ones(sz_test);
X_norm_test = [o_1_test,X_norm_test];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Linear Regression for TRAIN and Test
[W_out_lr, Y_train_pred_lr, MSE_train_lr, Y_test_pred_lr, MSE_test_lr] = ...
                Linear_regression_both(X_norm_train, Y_train_true, X_norm_test, Y_test_true);
            
Res_train(1,1) = str_res(1);
Res_train(1,2) = cellstr(num2str(MSE_train_lr));

Res_test(1,1) = str_res(1);
Res_test(1,2) = cellstr(num2str(MSE_test_lr));

%Ridge Regression for TRAIN and Test
W_train_rr = [];
for i =1:length(L)
    
    [W_out_rr, Y_train_pred_rr, MSE_train_rr, Y_test_pred_rr, MSE_test_rr] = ...
        Ridge_regression_both(X_norm_train, L(i), Y_train_true, X_norm_test, Y_test_true );

    Res_train(i+1,1) = str_res(i+1);
    Res_train(i+1,2) = cellstr(num2str(MSE_train_rr));
    Res_test(i+1,1) = str_res(i+1);
    Res_test(i+1,2) = cellstr(num2str(MSE_test_rr));
end

fprintf('_____________________________________________________________________\n');
fprintf('The results for Training Set using Linear and Ridge regression::\n');
fprintf('\t\tAlgorithm\t\t\t\t\tMSE\n')
disp(Res_train);

fprintf('_____________________________________________________________________\n');
fprintf('The results for Testing Set using Linear and Ridge regression::\n');
fprintf('\t\tAlgorithm\t\t\t\t\tMSE\n')
disp(Res_test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Ridge Regression with Cross-Validation
%10 fold cross validation
fprintf('\nIncrementing lamda by 0.05 after each iteration\nDisplaying only every 10th row for compactness');
fprintf('\n Results of Cross validation on Training Set::\nLamda Value \t MSE\n');
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
        
        [W_out_rr, Y_train_pred_rr, MSE_train_rr, Y_pred_test_rr, MSE_test_rr]= ...
            Ridge_regression_both(cv_X_norm_train, cv_lamda, cv_Y_train_true,cv_X_test,cv_Y_test_true);
              
%         if MSE_test_rr < min_mse
%             min_mse = MSE_test_rr;
%             w_min = W_out_rr;
%         end
        curr_mse_l = curr_mse_l + MSE_test_rr;
        
        
    end % end for
    Res_cv = [Res_cv; cv_lamda,(curr_mse_l/10)];
%     Res_cv_w = [Res_cv_w; cv_lamda,  w_min];
    if res_i ==1
        fprintf('%f\t\t%f\n',Res_cv(res_i,1),Res_cv(res_i,2));
    elseif mod(res_i,10)==0
        fprintf('%f\t\t%f\n',Res_cv(res_i,1),Res_cv(res_i,2));
    end
    res_i = res_i +1;
    %0.05 5.000100		28.671652
    %0.01 4.990100		28.671087
    cv_lamda = cv_lamda + .05;
end

%Find min value of MSE in Res_cv and use its corresponding value of Lamda


x = Res_cv(:,1);
y = Res_cv(:,2);

figure('Name','Lamda vs MSE')
xlabel('Lamda');
ylabel('MSE')
title('Ridge Regression with Cross Validation')
plot(Res_cv(:,1),Res_cv(:,2))
indexmin = find(min(y) == y);
xmin = x(indexmin);
ymin = y(indexmin);
strmin = sprintf('Min (%f, %f)',xmin,ymin);
text(xmin,ymin,strmin,'HorizontalAlignment','left');

fprintf('_____________________________________________________________________\n');
fprintf('\n Results of Cross validation on Testing Set::\n Lamda Value \t MSE\n');
[W_out_rr, Y_train_pred_rr, MSE_train_rr, Y_pred_test_rr, MSE_test_rr]= ...
            Ridge_regression_both(X_norm_train, xmin, Y_train_true, X_norm_test, Y_test_true);
 fprintf('%f\t\t%f\n',xmin,MSE_test_rr);
        

%FEATURE SELECTION
%BEST 4
A = [pcor(:,1),abs(pcor(:,2))];
sorted_pcor = sortrows(A,-2);
index_p = int16(sorted_pcor(1:4,1));
index_p = index_p +1;

X_fc_train = [X_norm_train(:,1),X_norm_train(:,index_p(1)),X_norm_train(:,index_p(2)),...
    X_norm_train(:,index_p(3)),X_norm_train(:,index_p(4))];

X_fc_test = [X_norm_test(:,1),X_norm_test(:,index_p(1)),X_norm_test(:,index_p(2)),...
    X_norm_test(:,index_p(3)),X_norm_test(:,index_p(4))];


[W_fc_train_lr, Y_pred_train, MSE_fc_lr_train, Y_fc_lr_test, MSE_fc_lr_test] =...
    Linear_regression_both(X_fc_train, Y_train_true,X_fc_test,Y_test_true );
fprintf('_____________________________________________________________________\n');
fprintf('\n\nFeatures Selected based on highest correlation are::\nAttrubute\t\tCorellation\n');
disp(sorted_pcor(1:4,:));
fprintf('\nMSE on training data:: %f',MSE_fc_lr_train);

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
    X_bf_train = [X_norm_train(:,1),X_norm_train(:,curr_cols(1)),X_norm_train(:,curr_cols(2)),...
        X_norm_train(:,curr_cols(3)),X_norm_train(:,curr_cols(4))];
    
    X_bf_test = [X_norm_test(:,1),X_norm_test(:,curr_cols(1)),X_norm_test(:,curr_cols(2)),...
        X_norm_test(:,curr_cols(3)),X_norm_test(:,curr_cols(4))];
    
    [W_bf_train_lr, Y_pred_train, MSE_bf_lr_train, Y_bf_lr_test, MSE_bf_lr_test] = ...
        Linear_regression_both(X_bf_train, Y_train_true, X_bf_test, Y_test_true );
        
    if  MSE_bf_lr_train < bf_min_mse_train
        bf_min_mse_train = MSE_bf_lr_train;
        bf_min_mse_train_test = MSE_bf_lr_test;
        bf_min_cols_train = curr_cols;
    end
         
    if MSE_bf_lr_test < bf_min_mse_test 
        bf_min_mse_test = MSE_bf_lr_test;
        bf_min_mse_test_train = MSE_bf_lr_train;
        bf_min_cols_test = curr_cols;
    end
    
end
fprintf('\nBest Columns for MIN MSE: %f on Training SET\n',bf_min_mse_train);
disp(bf_min_cols_train-1);
fprintf('Corresponding value of MSE: %f on Tesing SET\n',bf_min_mse_train_test);
fprintf('\nBest Columns for MIN MSE: %f on Testing SET\n',bf_min_mse_test);
disp(bf_min_cols_test-1);
fprintf('Corresponding value of MSE: %f on Training SET\n',bf_min_mse_test_train);

%Polynomial Feature Expansion
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

[W_train_lr, Y_pred_train, MSE_lr_train, Y_lr_test, MSE_lr_test] =...
    Linear_regression_both(X_pf_train, Y_train_true, X_pf_test, Y_test_true);

fprintf('\nMSE on Training data:: %f\n',MSE_lr_train);

fprintf('\nMSE on Testing data:: %f\n',MSE_lr_test);

% [B, FitInfo] = lasso(X_norm_train,Y_train_true,'CV',10);
% lassoPlot(B,FitInfo,'PlotType','CV');
