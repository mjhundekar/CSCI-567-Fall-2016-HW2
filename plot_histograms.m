function [pcor] = plot_histograms(o_train, col_name)
%plot_histograms Summary of this function goes here
%   Detailed explanation goes here
pcor = [];
for i = 1:length(o_train(1,:))-1
%     co_var = cov(o_train(:,i),o_train(:,14));
%     co_var_x_y = co_var(1,2);
%     std_x = std(o_train(:,i));
%     std_y = std(o_train(:,14));
%     pcor_i = co_var_x_y / (std_x*std_y);
%     pcor = [pcor;i,corr(o_train(:,i),o_train(:,14))];
    pcor_i = Pearson_correlation(o_train(:,i),o_train(:,14));
    pcor = [pcor;i,pcor_i];
    f =  figure('Name',col_name{i});
    histogram(o_train(:,i),10);
    title(sprintf('Histogram of %s distribution\nPearson Correlation = %f', col_name{i}, pcor(i,2)));
    xlabel(col_name{i});
    ylabel('Count');
    s = sprintf('%s.jpg',col_name{i});
    
    %     saveas(f,s);
end

end

