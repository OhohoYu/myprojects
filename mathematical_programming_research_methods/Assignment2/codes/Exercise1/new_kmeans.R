source("./squared_distance.R")
source("./choose_centroids.R")

NewKmeans <- function(X, n_clusters){
  # X is a matrix (represented by a Data Frame) R^n where each row represents 
  # a data point
  # n_clusters is the number of clusters we want to divide the data in.
  m <- nrow(X)
  n <- ncol(X)
  # create matrix to keep track of which data point belongs to which cluster
  clustA <- matrix(0L, nrow = m, ncol = 2)
  # choose centers randomly   
  centroids <- randCentroids(X, n_clusters)
  # keep the original for plotting in order to test
  original <- centroids
  
  # has_changed is the boolean variable that keeps track of the changes in any of the points
  has_changed <- TRUE
  # update the clusters that a data point belongs to
  # the algorithm ends where no more points are updated, has_changed = FALSE
  while(has_changed){
    has_changed <- FALSE
    for(i in 1:m){
      # my_distance, set the distance to inf, the first time it will be less than any distance calculated
      # keeps the distance of the Ith point to the Jth centroid
      my_distance <- 50000
      my_new_index <- -1
      # my_new_index is the variable that updates the index of the cluster that the
      # ith observation belongs to
      for(j in 1:n_clusters){
        # print(X[i,])
        # print(centroids[j,])
        cal_dist <- sqrt(sum((X[i,] - centroids[j,])^2))
        
        if(cal_dist < my_distance){
          my_distance <- cal_dist
          my_new_index <- j
        }
      }
      
      if(clustA[i,1] != my_new_index){
        has_changed <- TRUE
      }
      
      clustA[i,1] <- my_new_index
      clustA[i,2] <- cal_dist^2
    }
    # calculate the new centroids of the clusters
    for(cent in 1:n_clusters){
      if(length(which(clustA[,1] == cent)) == 0){
        for(colm in 1:n){
          minJ <- min(X[,colm])
          maxJ <- max(X[,colm])
          rangeJ <- maxJ - minJ
          centroids[cent,colm] <- minJ + rangeJ * runif(1)
        }
      }else{
        clust_points <- X[which(clustA[,1] == cent),]
        for(colm2 in 1:ncol(X)){
          centroids[cent,colm2] = mean(clust_points[,colm2]) 
        }
      }
    }
  }
  
  # create a data.frame with the values to return
  df <- data.frame(Original = original, Centroids = centroids)
  # return the data.frame with the Original centroids and the Centroids that resulted from the algorithm
  return(clustA)
}