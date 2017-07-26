clear all; 
% set random seed
rng(332) 
k=3;  % cluster number
% partition limits
part2data_original = load('iris.csv'); % data generation provided in question
% returns information on cluster assignments as cell array
% clusters.clusters composed data points assigned to each cluster 

% FOR 3 PRINCIPAL COMPONENTS
pca_no=3; % number of components for PCA function. 
part2data = my_pca(part2data_original,pca_no);
[clusters, tot_cost] = new_kmeans_clust(part2data, k); 
% retrieve individual cluster assignments
centroid1 = clusters.centroids{1};
centroid2 = clusters.centroids{2};
centroid3 = clusters.centroids{3};
cluster1 = clusters.clusters{1};
cluster2 = clusters.clusters{2};
cluster3 = clusters.clusters{3}; 
% PLOTS
figure
x1 =scatter3(cluster1(:,1),cluster1(:,2),cluster1(:,3),'s');
hold on 
x2 =scatter3(cluster2(:,1),cluster2(:,2),cluster2(:,3),'o');
x3 =scatter3(cluster3(:,1),cluster3(:,2),cluster3(:,3),'*');

x4 = scatter3(centroid1(:,1),centroid1(:,2),centroid1(:,3),375,[0.5,0,0],'+');
scatter3(centroid2(:,1),centroid2(:,2),centroid2(:,3),375,[0.5,0,0],'+')
scatter3(centroid3(:,1),centroid3(:,2),centroid3(:,3),375,[0.5,0,0],'+')
hold off
legend([x1 x2 x3 x4],'Cluster 1', 'Cluster 2', 'Cluster 3', 'Init. Centroids')

