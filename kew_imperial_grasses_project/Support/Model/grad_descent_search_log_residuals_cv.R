################################################################################           
#                                                                              #
#                                                                              #
# This function is an altered implementation of the sister function            #
# grad_descent_search_log_residuals. Here the script runs the same algorithm   # 
# as its sister. Here the algorithm is applied repeatedly to the data with one #
# year left out and the results are compiled and output as a .csv file. The    #
# parameters function in the same way as laid out in the sister script.        #
#                                                                              #
# To use simply edit the variables below. The output will be three graphs,     #
# a summary csv of the model parameters and scores and a csv of the raw data   #
# along with the predicted values.                                             #
#                                                                              #
# Antonio Remiro, Jonathan Williams, 2016                                      #
# remiroantonio@gmail.com                                                      #
# jonvw28@gmail.com                                                            #        
#                                                                              #
################################################################################
#
########################## EXPLANATION OF ARGUMENTS ############################
#
# spec.data - data frame where column 1 is start year, column 2 is number of new
# species recorded in the time window, column 3 is the cumulative number of 
# species up to the given window.
#
# tax.data - data frame where column 1 is the start year and column 2 is the is 
# number of active taxonomists in the time window.
#
# en.yr - year at which data ends so as to enable trimming if need be
# eg 2015
#
# mult - multiple of current total species to start at as maxmimum guess for St
# eg 3
#
# guess.n - Guesses per round for St values
# eg 500
#
# ratio - Ratio of top scoring guesses to keep from all guesses per round
# eg 0.2
#
# stretch - Range Stretch to apply at each end (ie 1.25 would mean extend the
# range in each iteration by 25%)
# eg 1.5
#
# max.it - Max iteratations of guessing St
# eg 20
#
# scale - Scaling to apply to taxonomist numbers and species numbers respectively
# (years are dealt with automatically) - This is to help gradient descent
# efficiency
# eg c(100,1000)
#
# rng.a, rng.b - Range to test for a and b starting point in each gradient 
# descent - note these are not transformed by the scalings (ie these values will
#  be used as they currently are directly with the tranformed data, however a 
# and b as output are for the none scaled data) 
# eg
# rng.a <- c(-0.1,0.1)
# rng.b <- c(-0.1,0.1)
#
# ab.guesses - No of initial values of a and b to try respectively
# eg c(100,100)
#
# max.grad - Max repetitions of grad descent to get a,b for each St value
# eg 500
#
# alpha - Step size for each gradient descent step
# eg 0.01
#
# min.alp - Minimum step size - gradient descent stops if a step smaller than 
# this is required
# eg 2e-14
#
# grd.rat - Ratio for gradient/parameter value where gradient descent should be 
# terminated - ie once this ratio is reached, gradient descent ends
# eg 1e-4
#
# out.dir - Directory where the output directory should go 
# eg "./Output"
#
# id.str - Identifier string - included in the file names and as subdirectory
# eg "grass_1755_5y"
#
# mod.dir - sub-directory where the model data should go
# eg "regression_search/cross_validation"
#
#
#
grad_descent_search_log_residuals_cv <- function(spec.data, tax.data, en.yr, mult, 
                                                guess.n, ratio, stretch, max.it,
                                                scale, rng.a, rng.b, ab.guesses,
                                                max.grad, alpha, min.alp,
                                                grad.rat, out.dir, id.str, 
                                                mod.dir) {
        #
        # Check for directory and create if needed
        #
        tmp.dir <- paste(out.dir,"/",id.str,"/",mod.dir,sep = "")
        if(dir.exists(tmp.dir)==FALSE){
                dir.create(tmp.dir,recursive = T)
        }
        rm(out.dir)
        #
        # Install any dependancies and load functions
        #
        source("./kew_grasses/Support/packages.R")
        source("./kew_grasses/Support/functions.R")
        #
        #
        ########################### DATA PROCESSING ####################################        #
        # Merge data
        #
        data <- table.merge(spec.data,tax.data,data.index=2,split = 3)
        rm(spec.data,tax.data)
        #
        # Tidy data and remove any partial end year (ie if the final window is shorter 
        # than the other windows then it is excluded)
        # note: years in data are start years
        #
        if(!replicate.research) {
                yr.int <- data[2,1] - data[1,1]
                if((en.yr-data[1,1]+1) %% yr.int != 0) {
                        data <- data[1:(nrow(data)-1),]
                }
        }
        rm(en.yr)
        #
        # Tidy data and remove any partial end year (ie if the final window is shorter 
        # than the other windows then it is excluded)
        # note: years in data are start years
        #
        if(minimum.cumulative) {
                data <- dplyr :: filter(data, data[,3] >= min.c)
        }
        #
        # Scale data so as to make convergence of gradient descent better
        # Years are scaled so the first year is 0 and the final year is 1
        #
        scale <- c(scale,yr.int,data[1,1])
        names(scale) <- c("taxon","species","year_gap","start_year")
        data[,4] <- data[,4]/scale[1]
        data[,2:3] <- data[,2:3]/scale[2]
        data[,1] <- (data[,1]-data[1,1])/(yr.int*(nrow(data)-1))
        rm(yr.int)
        #
        # Set up results variable to store results of each round of jack-knife
        #
        cv.results <- matrix(NA,nrow = nrow(data),ncol = 10+nrow(data))
        cv.results <- as.data.frame(cv.results)
        names(cv.results) <- c("year_out","a","b","St","cost_fn","iterations_taken",
                               "guesses_per_it","ratio_kept","expansion_per_it",
                               "initial_multiple",as.character(data[,1]))
        #
        # Apply Jack-knifing
        #
        for(j in 1:nrow(data)) {
                tmp.data <- data[-j,]
                #
                # Counter for human convenience
                #
                cat("Jack-Knife ",j,"\n")
                #
                #
                ########################## Optimization Algorithm ######################
                #
                # Calculate initial guesses for a and b for each grid search - these remain the 
                # same across all guesses of St
                #
                a.guess <- seq(rng.a[1],rng.a[2],length = ab.guesses[1])
                b.guess <- seq(rng.b[1],rng.b[2],length = ab.guesses[2])
                #
                # Calculate the current level of total species
                #
                start <- data[nrow(tmp.data),2] + data[nrow(tmp.data),3]
                #
                # Pick initial guesses as starting with the above and ending with the multiple
                # of this as given in the parameters, using equally spaced guesses as set
                #
                guesses <- seq(start+0.001,mult*start,length.out = guess.n)
        
                ###########################################################################
                #
                # Set flag as counter for number of iterations
                # Set mark as a score for how big the range of candidate values for St is. 
                # Once this is below 0.5 we know we have convergence to the precision
                # of 1 integer.
                #
                flag <- 0
                mark <- 2/scale[2]
                #
                # Iteration over each set of guesses starts here and ends when maximum 
                # iterations are reached, or convergence as defined abpove is reached
                #
                while (mark > 0.5/scale[2] && flag < max.it) {
                        #
                        # Counter for human convenience
                        #
                        cat("Iteration ",flag+1,"\n")
                        #
                        # Create placeholder for error score of each guess for St, along with
                        # the best fitting values of a and b and the number of gradient descent
                        # steps taken
                        #
                        results <- matrix(0,length(guesses),ncol = 5)
                        #
                        # Find best choice of a, b for each guess of St via grid search and
                        # steepest descent
                        #
                        for (i in 1:length(guesses)) {
                                #
                                # Create cached variable for faster computation
                                #
                                grad.cache <- matrix(0,nrow(tmp.data),2)
                                grad.cache[,1] <- (guesses[i]-tmp.data[,3])*tmp.data[,4]
                                grad.cache[,2] <- log(tmp.data[,2])
                                #
                                # Calculate best place to start gradient descent via grid 
                                # search
                                #
                                init.score <- matrix(0,nrow = ab.guesses[1],ncol=ab.guesses[2])
                                for(a in 1:length(a.guess)){
                                        for(b in 1:length(b.guess)){
                                                init.score[a,b] <- joppa.cost(tmp.data,a.guess[a],
                                                                        b.guess[b],
                                                                        guesses[i],
                                                                        T,grad.cache[,1])
                                        }
                                }
                                rm(a,b)
                                #
                                # Pick best fitting a and b as initial guesses for grad descent
                                #
                                st.ind <- which(init.score[,]==min(init.score[,],na.rm = T)
                                        ,arr.ind = T)
                                rm(init.score)
                                #
                                # Apply gradient descent with fixed St - ie for a,b
                                #
                                cur.par <- c(a.guess[st.ind[1,1]],b.guess[st.ind[1,2]],
                                         guesses[i])
                                rm(st.ind)
                                grad <- joppa.grad(tmp.data,cur.par[1],cur.par[2],cur.par[3],T,
                                           grad.cache)
                                #
                                # Create a flag to count iterations of gradient descent
                                #
                                grad.flag <- 0
                                while (grad.flag < max.grad) {
                                        #
                                        # Create flag for whether adaptive step size has been 
                                        # used
                                        #
                                        alp.flag <- 0
                                        grd.sze.flag <- 0
                                        nxt.par <- cur.par
                                        #
                                        # Calculate next paramter choice using gradient descent
                                        # step and calculate gradient here
                                        #
                                        nxt.par[1] <- cur.par[1] - grad[1]*alpha
                                        nxt.par[2] <- cur.par[2] - grad[2]*alpha
                                        nxt.grad <- joppa.grad(tmp.data,nxt.par[1],nxt.par[2],nxt.par[3],
                                                                T,grad.cache)
                                        #
                                        # Now consider if adaptive step size is needed:
                                        #
                                        tmp.a <- alpha
                                        small.grad <- grad
                                        #
                                        # First ensure step taken doesn't cause issue with 
                                        # non-defined logs
                                        # If the step will cause predictions of negative species
                                        # discovery (causing issues for taking logs) then the 
                                        # effective gradient will be halved until this issue is
                                        # resolved
                                        #
                                        while(is.na(nxt.grad[1]) || is.na(nxt.grad[2])){
                                                if(min(abs(small.grad)) < min.alp) {
                                                        grd.sze.flag <- 2
                                                        break
                                                }
                                                small.grad = small.grad/2
                                                tmp.grad <- nxt.grad
                                                tmp.par <- cur.par
                                                tmp.par[1] <- cur.par[1] - small.grad[1]*tmp.a
                                                tmp.par[2] <- cur.par[2] - small.grad[2]*tmp.a
                                                nxt.grad <- joppa.grad(tmp.data,tmp.par[1],tmp.par[2],
                                                                        tmp.par[3],T,grad.cache)
                                        }
                                        while(min(nxt.grad*grad) < 0 && grd.sze.flag != 2) {
                                                if(tmp.a < min.alp) {
                                                        alp.flag <- 2
                                                        break
                                                }
                                                #
                                                # Comes before halving as the eventual step size 
                                                # is double the one that ends this loop. Hence
                                                # can try a step smaller than minimum step size
                                                # as long as the resulting doubled step would
                                                # still be big enough
                                                #
                                                tmp.a = tmp.a/2
                                                tmp.grad <- nxt.grad
                                                tmp.par <- cur.par
                                                tmp.par[1] <- cur.par[1] - small.grad[1]*tmp.a
                                                tmp.par[2] <- cur.par[2] - small.grad[2]*tmp.a
                                                nxt.grad <- joppa.grad(tmp.data,tmp.par[1],tmp.par[2],
                                                               tmp.par[3],T,grad.cache)
                                                alp.flag <- 1
                                        }
                                        #
                                        # In case that step size is too small, revert to most 
                                        # recent accepted parameters and break the loop
                                        #
                                        if(alp.flag == 2) {
                                                break
                                        }
                                        if(grd.sze.flag == 2) {
                                                break
                                        }
                                        #
                                        # If alpha modified in a permissible way then take the
                                        # related step
                                        #
                                        if(alp.flag == 1){
                                                tmp.a = 2*tmp.a
                                                nxt.par <- cur.par
                                                nxt.par[1] <- cur.par[1] - small.grad[1]*tmp.a
                                                nxt.par[2] <- cur.par[2] - small.grad[2]*tmp.a
                                                nxt.grad <- tmp.grad
                                                rm(tmp.par,tmp.grad)
                                        }
                                        #
                                        # Pass on the parameters for the next iteration
                                        #
                                        grad <- nxt.grad
                                        cur.par <- nxt.par
                                        rm(nxt.par,nxt.grad,tmp.a,small.grad)
                                        #
                                        # Test if gradient is now sufficiently small relative to
                                        # the parameters and if so end the gradient descent
                                        #
                                        test <- abs(grad/cur.par[1:2])
                                        if(test[1]< grd.rat && test[2] < grd.rat){
                                                rm(test)
                                                break
                                        }
                                        rm(test)
                                        grad.flag <- grad.flag + 1
                                        #
                                        # Output warning if the maximum number of gradient 
                                        # descent steps are taken
                                        #
                                        rm(alp.flag)
                                        #
                                        # Calculate cost function at the paramter values determined
                                        #
                                        results[i,1] <- joppa.cost(tmp.data,cur.par[1],cur.par[2],
                                                                   guesses[i],T,grad.cache[,1])
                                        results[i,2:4] <- cur.par
                                        results[i,5] <- grad.flag
                                        #
                                        # Counter for human to see progress
                                        #
                                }
                                rm(cur.par, grad, grad.cache,grad.flag)
                                if((i*100/guess.n)%%10==0) {
                                        if(i/guess.n == 1) {
                                                cat(i*100/guess.n,"% complete!\n")   
                                        }else{
                                                cat(i*100/guess.n,"% complete...") 
                                        }
                                }
                        }
                        rm(i)
                        #
                        # Order the scores for each round of guesses and select only the top
                        # proportion, as set by the ratio parameter
                        #
                        picks <- guesses[order(results[,1])[1:(ratio*length(guesses))]]
                        #
                        # Calculate the range of these selected values and extend it by the
                        # stretch factor set in the parameters
                        #
                        rng <- range(picks)
                        extra <- (rng[2]-rng[1])*(stretch-1)/2
                        rng[1] <- rng[1] - extra
                        rng[2] <- rng[2] + extra
                        #
                        # Ensure the range never drops below the current total number of species
                        #
                        if(rng[1] <= start){
                                rng[1] <- start + 1
                        }
                        #
                        # Use this range to pick the new guesses for the next iteration
                        #
                        guesses <- seq(rng[1],rng[2],length.out = guess.n)
                        #
                        # Score current convergence
                        #
                        mark <- rng[2]-rng[1]
                        rm(rng,extra)
                        #
                        # Cache initial guesses and their costs for later graphing
                        #
                        if(flag == 0){
                                res.cache <- results
                        }
                        #
                        flag <- flag + 1
                }
                rm(guesses,results)
                #
                # Transform all of data back into more meaningful form
                #
                tmp.data[,4] <- tmp.data[,4]*scale[1]
                tmp.data[,2:3] <- tmp.data[,2:3]*scale[2]
                tmp.data[,1] <- tmp.data[,1]*scale[3]*(nrow(tmp.data)-1) + scale[4]
                #
                # Transform parameters back into corrected form
                #
                params[3] <- params[3]*scale[2]
                params[1] = (params[1]-params[2]*scale[4]/(scale[3]*(nrow(data)-1)))/scale[1]
                params[2] = params[2]/(scale[1]*scale[3]*(nrow(data)-1))
                #
                # Calculate a and b for the best choice of St output and store these
                #
                cv.results[j,2:4]<- params
                cv.results[j,1] <- data[j,1]
                rm(test,weight,picks)
                cv.results[j,5] <- joppa.cost(tmp.data,params[1],params[2],params[3])
                cv.results[j,6] <- flag
                cv.results[j,7] <- guess.n
                cv.results[j,8] <- ratio
                cv.results[j,9] <- stretch
                cv.results[j,10] <- mult
                if(mark > 0.5) {
                        cat("Algorithm failed to converge to a value of total species accurate",
                        "to the nearest integer after",max.it,"iterations. Try using more",
                        "iterations or reducing the ratio of values passed on after each",
                        "round\n")
                } 
                rm(mark, flag, start)
                tmp <- (rep(params[3],nrow(tmp.data))-tmp.data[,3])
                pred <- (params[1] + params[2]*tmp.data[,1])*tmp.data[,4]*tmp
                if(j>1){
                        for(k in 1:(j-1)){
                                cv.results[j,k+10] <- pred[k]
                        }
                        rm(k)
                }
                if(j < nrow(data)){
                        for(l in j:nrow(tmp.data)){
                                cv.results[j,l+11] <- pred[l]
                        }
                        rm(l)
                }
                rm(tmp,tmp.data,pred)
                cat("Jack-Knife ",j," complete!\n")
        }
        write.csv(cv.results,file=paste(tmp.dir,id.str,"_grad_descent_search_cv.csv",
                                        sep=""),
                  row.names = FALSE)
        rm(mult,stretch,max.it,ratio,guess.n,j,params)
        rm(data,tmp.dir,id.str,cv.results)
        rm(a.guess,b.guess, ab.guesses, min.alp, grd.rat, alpha, max.grad, rng.a,
           rng.b, scale)
}