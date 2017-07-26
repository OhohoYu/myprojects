rm(list=ls(all=TRUE))

if(!require("combinat")) {
  install.packages("combinat")        
}
library(combinat)

# set fixed random seed in order to replicate results
set.seed(8)

A1 <- t(matrix(c(0.5, 0.2, 0, 2), ncol = 2));
u1 <- c(4, 0);

A2 <- t(matrix(c(0.5, 0.2, 0, 0.3), ncol = 2));
u2 <- c(5, 7);

A3 <- t(matrix(c(0.8, 0, 0, 0.8), ncol = 2));
u3 <- c(7, 4);

data <- matrix(rnorm(150*2,mean=0,sd=1), 150, 2) 

# test algorithm in 2D dataset where each dataset has a different
# mean and variance

# Group 1 (S1)
for(f in 1:50){
  data[f,] <- u1 + A1 %*% data[f,]
}

# Group 2 (S2)
for(s in 51:100){
  data[s,] <- u2 + A2 %*% data[s,]
}

# Group 3 (S3)
for(t in 101:150){
  data[t,] <- u3 + A3 %*% data[t,]
}

# data <- readMat('data.mat')
# my_data <- data[[1]]
#

# K-means algorithm testing 
source("./new_kmeans.R")
source("./new_kmeans2.R")
my_data <- data
kmm <- NewKmeans(my_data, 3)
df <- NewKmeans2(my_data,3)


plot(my_data, xlab="x", ylab="y")
points(my_data[which(kmm[,1] == 1),], col = 'darkgreen', pch = 15)
points(my_data[which(kmm[,1] == 2),], col = 'darkorchid', pch = 16)
points(my_data[which(kmm[,1] == 3),], col = 'firebrick', pch = 17)
# # # Plot the original random centroids
points(df[c('Original.1', 'Original.2')], col = 'blue', pch=4, cex=5)
# # # Plot the re-calculated centroids
points(df[c('Centroids.1', 'Centroids.2')], col = 'black', pch = 10,cex=5, bg='red')
legend ("bottomright", bty = "n", inset = c(0.1,0),legend = c('C1','C2', 'C3', 'Orig. centroids',
        'New centroids'),cex=1.2,col = c("darkgreen", "firebrick", "darkorchid", "blue","black"),
                                                    pch = c(15, 17, 16,4,10))
