clear all; 
% set random seed
rng(332) 
k=3;  % cluster number
tot_iters = 100; % total iterations. 
% initialise occe place-keeper
occes= zeros(1,tot_iters); % occe placekeeper
% partition limits
low_lim = 50;
mid_lim = 100;
high_lim =150;
% call Kmeans 
for i = 1:tot_iters;
    part2data = genData2; % data generation provided in question
    % returns information on cluster assignments as cell array
    % clusters.clusters composed data points assigned to each cluster  
    clusters = new_kmeans(part2data, k); 
    % clusters.clusters composed of data points assigned to each cluster
    % clusters.centroids composed of centroid information. 
    true_splits{1} = part2data(1:low_lim,:);
    true_splits{2} = part2data(low_lim+1:mid_lim,:);
    true_splits{3} = part2data(mid_lim+1:high_lim,:);
    % occes computation -> assigned clusters vs true splits
    occes(i) = compute_occe(clusters.clusters,true_splits);
end
occe_mean = sum(occes)/length(occes);
occe_stdev = std(occes); % standard deviation of occe

fprintf('Mean after 100 trials: %.3f \n', occe_mean)
fprintf('Standard deviation after 100 trials: %.3f \n', occe_stdev)