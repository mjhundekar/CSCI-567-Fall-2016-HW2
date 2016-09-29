function [ W_out, Y_pred ] = Linear_regression( X_in, Y_in )
%Linear_regression Summary of this function goes here
%   Detailed explanation goes here
X_t = transpose(X_in);
X_X_t = X_t * X_in;
% X_X_t_inv = inv(X_X_t);
W_out = X_X_t \ X_t * Y_in;

W_out = transpose(W_out);
% Replace inv(A)*b with A\b
% Replace b*inv(A) with b/A
Y = [];
for i =1:length(X_in(:,1))
    temp_y = W_out * transpose(X_in(i,:));
    Y = [Y;temp_y];
end
% Should not be transpose as W_out[14,1] X_in[433,14]
% Y = transpose(W_out) * X_in;
% Y_pred = sum(Y);

Y_pred = Y;
end

