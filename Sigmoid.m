function [ sigma ] = Sigmoid( A )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

sigma = zeros(length(A),'double');
sigma = 1.0 ./ (1.0 + exp(-A));

end

