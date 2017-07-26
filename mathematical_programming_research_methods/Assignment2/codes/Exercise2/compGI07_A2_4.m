clear all; 
% set random seed
rng(332) 
k=3;  % cluster number
% partition limits
part2data_original = load('iris.csv'); % data generation provided in question
% returns information on cluster assignments as cell array
% clusters.clusters composed data points assigned to each cluster 

% FOR 2 PRINCIPAL COMPONENTS
pca_no=2; % number of components for PCA function. 
part2data = my_pca(part2data_original,pca_no);
[clusters, tot_cost] = new_kmeans_clust(part2data, k); 
% retrieve individual cluster assignments
centroid1 = clusters.centroids{1};
centroid2 = clusters.centroids{2};
centroid3 = clusters.centroids{3};
cluster1 = clusters.clusters{1};
cluster2 = clusters.clusters{2};
cluster3 = clusters.clusters{3}; 
figure
x1=scatter(cluster1(:,1),cluster1(:,2),'m','o');
hold on
x2=scatter(cluster2(:,1),cluster2(:,2),'b','x');
x3=scatter(cluster3(:,1),cluster3(:,2),'y','s');
x4=scatter(centroid1(:,1),centroid1(:,2),375,[0.5,0,0],'+');
scatter(centroid2(:,1),centroid2(:,2),375,[0,0.5,0],'+');
scatter(centroid3(:,1),centroid3(:,2),375,[0,0,0.5],'+');
hold off

legend([x1 x2 x3 x4],'Cluster 1', 'Cluster 2', 'Cluster 3', 'Init. Centroids')


