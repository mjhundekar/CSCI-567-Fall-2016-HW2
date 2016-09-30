function [ Y_t ] = Predict_Y( W, X )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
Y_t = [];
for i =1:length(X(:,1))
    temp_y = W * transpose(X(i,:));
    Y_t = [Y_t;temp_y];
end
Y = Y_t;

end

