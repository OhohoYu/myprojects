randCentroids <- function(X, n_centroids){
  # Return n_centroids amount of centroids from the Data Frame X
  
  # initialize an empty matrix
  centroids <- matrix(0, n_centroids, ncol(X))
  
  # create as many centroids as clusters we want to have in our algorithm
  # for(i in 1:n_centroids){
  #   # initialize the random points at each iteration
  #   cent <- NULL
  #   # select k random points
  #   cent <- (X[sample(nrow(X), n_centroids),])
  #   # store the x and y values of each centroid in the array
  #   for(j in 1:ncol(X)){
  #     centroids[i,j] <- sum(cent[,j])/n_centroids
  #   }
  # }
  
  n <- ncol(X)
  for(j in 1:n){
    minJ <- min(X[,j])
    maxJ <- max(X[,j])
    rangeJ <- maxJ - minJ
    centroids[,j] <- minJ + rangeJ * runif(n_centroids)
    # print(centroids[,j])
    # print(centroids)
  }
  # return the array with all the x and y coordinates of the initial centroids
  return(centroids)
}