% The function new_kmeans.m models the k-means clustering algorithm. 
%
% It takes as inputs a set of datapoints X and the specified number of
% clusters one wishes to assign to the data. 
% 
% It ouputs a cell array cluster_info which contains the centroids of
% each cluster and the data point coordinates corresponding to each cluster.
%
% cluster_info.centroids - centroid information
% cluster_info.clusters - data points assigned to each cluster
%
function [cluster_info] = new_kmeans(X, n_clusters)
    [m,~]=size(X); % m is the number of data points
    % randperm to randomly choose centroids instead of choose_centroids.r
    centroids = X(randperm(m,n_clusters),:);
    % initialise distance metrics
    my_distance = 50000; % dist_provisional - dist
    dist = zeros(m,n_clusters);
    % update clusters that a data point belongs to. the algorithm
    % ends where no more points are updated, dist is not lesser than
    % dist_provisional
    while norm(my_distance) > 0 
        dist_provisional = dist; % distance metric for t-1
        dist = zeros(m, n_clusters);  % restart distance metric
        % track closest centroids
        for i = 1:m  % loop over each data point
            % recall centroids is coord vector of cluster centers
            [m_cent,~] = size(centroids);
            metric = ones(1, m_cent); % initialisation
            for j = 1:m_cent % loop over each centroid
                % euclidean distance between point and centroid
                metric(j) = norm(X(i,:) - centroids(j,:)); 
            end
            % assigned centroid to data point minimises distance
            [~, closest] = min(metric);
            dist(i, closest) = 1;
        end
        % calculate new cluster centroids    
        for j = 1:n_clusters % for each cluster
            % create a matrix to keep track of which data point belongs to
            % each cluster
            matrix = X(dist(:,j) == 1,:); 
            [mmat, ~] = size(matrix);
            centroids(j,:) = sum(matrix)/mmat; 
        end
        my_distance = dist_provisional - dist;
    end
    % a cell array cluster_info is returned with coordinate information of
    % the centroids for each cluster and the data point coordinates
    % corresponding to each cluster.
    for j = 1:n_clusters
        cluster_info.centroids{j} = centroids(j,:); 
        cluster_info.clusters{j} = X(dist(:,j) == 1,:);
    end
end    