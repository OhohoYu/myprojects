clear all; 
% set random seed
rng(332) 
k=3;  % cluster number
pca_no=1; % number of components for PCA function. 
tot_iters = 100; % total iterations. 
% initialise occe place-keeper
occes= zeros(1,tot_iters); % occe placekeeper
% partition limits
low_lim = 50;
mid_lim = 100;
high_lim =150;
tot_tot_cost = zeros(1,tot_iters); % keeps track of cost during all iterations  
% call Kmeans 
for i = 1:tot_iters;
    part2data = load('iris.csv'); % data generation provided in question
    % returns information on cluster assignments as cell array
    % clusters.clusters composed data points assigned to each cluster 
    part2data = my_pca(part2data,pca_no);
    [clusters, tot_cost] = new_kmeans_clust(part2data, k); 
    % clusters.clusters composed of data points assigned to each cluster
    % clusters.centroids composed of centroid information. 
    tot_tot_cost(i) = tot_cost;
    true_splits{1} = part2data(1:low_lim,:);
    true_splits{2} = part2data(low_lim+1:mid_lim,:);
    true_splits{3} = part2data(mid_lim+1:high_lim,:);
    % occes computation -> assigned clusters vs true splits
    occes(i) = compute_occe(clusters.clusters,true_splits);
end
occe_mean = sum(occes)/length(occes);
occe_stdev = std(occes); % standard deviation of occe
mean_cost = mean(tot_tot_cost);
std_cost = std(tot_tot_cost);
occes_temp =  sort(occes,'ascend');
occes_low3 = occes_temp(1:3);
cost_temp = sort(tot_tot_cost, 'ascend');
cost_low3 = cost_temp(1:3);


fprintf('Mean after 100 trials: %.3f \n', occe_mean)
fprintf('Standard deviation after 100 trials: %.3f \n', occe_stdev)
fprintf('Number of principal components: %d \n', pca_no)
fprintf('Mean optimum cost is: %.3f \n', mean_cost)
fprintf('Standard deviation of optimum cost is: %.3f \n', std_cost)
fprintf('Lowest three occes are: ')
disp(occes_low3)
fprintf('Corresponding lowest three costs are: ')
disp(cost_low3)
% bar chart of occes to rank
b = bar(occes_temp);

