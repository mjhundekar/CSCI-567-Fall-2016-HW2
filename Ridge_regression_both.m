function [ W_out, Y_train_pred, MSE_train, Y_test_pred, MSE_test ] = Ridge_regression_both( X_train, Lamda, Y_train, X_test, Y_test )
%Ridge_regression Summary of this function goes here
%   Detailed explanation goes here

%learn W from training data
X_t = transpose(X_train);
X_t_X = X_t * X_train;
sz = size(X_t_X);
L_I = Lamda * eye(sz);

X_t_X_LI_inv = pinv(X_t_X + L_I);
W_out = X_t_X_LI_inv * X_t * Y_train;

W_out = transpose(W_out);

%Predict Y_train
Y_train_pred = Predict_Y(W_out, X_train);

%Calculate MSE for Y_train
MSE_train = ( sum( (Y_train - Y_train_pred).^2) ) / length(X_train(:,1));


%Predit Y for Test Data
Y_test_pred = Predict_Y(W_out,X_test);

%Calculate MSE for Y_test
MSE_test = ( sum( (Y_test - Y_test_pred).^2) ) / length(X_test(:,1));

% Should not be transpose as W_out[14,1] X_in[433,14]
% Y = transpose(W_out) * X_in;
% Y_pred = sum(Y);

end

