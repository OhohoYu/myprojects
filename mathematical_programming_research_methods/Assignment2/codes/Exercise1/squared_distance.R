squaredDistance <- function(vec1, vec2){
  # squaredDistance calculates the norm2 distance
  
  # calculate the squared distance between two vectors
  return(sqrt(sum(vec1-vec2)^2))
}