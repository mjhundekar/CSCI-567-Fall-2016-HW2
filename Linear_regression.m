function [ W_out, Y_pred ] = Linear_regression( X_in, Y_in )
%Linear_regression Summary of this function goes here
%   Detailed explanation goes here
X_t = transpose(X_in);
X_X_t = X_t * X_in;
% X_X_t_inv = inv(X_X_t);
W_out = X_X_t \ X_t * Y_in;

% Replace inv(A)*b with A\b
% Replace b*inv(A) with b/A

Y = W_out * X_in;
Y_pred = sum(Y);


end

