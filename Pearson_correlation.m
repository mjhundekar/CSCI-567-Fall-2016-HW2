function [ pcor_i ] = Pearson_correlation( X, Y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

co_var = cov(X,Y);
co_var_x_y = co_var(1,2);
std_x = std(X);
std_y = std(Y);
pcor_i = co_var_x_y / (std_x*std_y);
end

