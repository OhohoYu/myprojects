# 8 parameters, piRew & piPun are greater than 0 with exp.
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  real outcome[N, T];
  int<lower=0, upper=1> pressed[N, T];
  int<lower=1, upper=4> cue[N, T];
}
transformed data {
  vector[4] initV;
  initV  = rep_vector(0.0, 4);
}
parameters {
  # declare as vectors for vectorizing
  vector[8] mu_p;  
  vector<lower=0>[8] sigma; 
  vector[N] xi_pr;         # noise 
  vector[N] epRew_pr;      # learning rate - reward
  vector[N] epPun_pr;      # learning rate - punishment
  vector[N] b_pr;          # go bias 
  vector[N] piRew_pr;      # pavlovian bias - reward
  vector[N] piPun_pr;      # pavlovian bias - punishment
  vector[N] rhoRew_pr;     # rho reward, inv temp
  vector[N] rhoPun_pr;     # rho punishment, inv temp 
}
transformed parameters{
  vector<lower=0,upper=1>[N] xi;
  vector<lower=0,upper=1>[N] epRew;
  vector<lower=0,upper=1>[N] epPun;
  vector[N] b; 
  vector<lower=0>[N] piRew; 
  vector<lower=0>[N] piPun; 
  vector<lower=0>[N] rhoRew;
  vector<lower=0>[N] rhoPun;
     
  for (i in 1:N) {
    xi[i]     = Phi_approx( mu_p[1] + sigma[1] * xi_pr[i] );
    epRew[i]  = Phi_approx( mu_p[2] + sigma[2] * epRew_pr[i] );
    epPun[i]  = Phi_approx( mu_p[3] + sigma[3] * epPun_pr[i] );
  }
  b         = mu_p[4] + sigma[4] * b_pr; # vectorization
  piRew     = exp( mu_p[5] + sigma[5] * piRew_pr );
  piPun     = exp( mu_p[6] + sigma[6] * piPun_pr );
  rhoRew    = exp( mu_p[7] + sigma[7] * rhoRew_pr );
  rhoPun    = exp( mu_p[8] + sigma[8] * rhoPun_pr );
}
model {  
  # gng_m5: RW(rew/pun) + noise + bias + pi(rew/pun) + separate learning (rew/pun) model 
  # hyper parameters
  mu_p  ~ normal(0, 1.0); 
  sigma ~ cauchy(0, 5.0);
  
  # individual parameters w/ Matt trick
  xi_pr     ~ normal(0, 1.0);   
  epRew_pr  ~ normal(0, 1.0);
  epPun_pr  ~ normal(0, 1.0);   
  b_pr      ~ normal(0, 1.0); 
  piRew_pr  ~ normal(0, 1.0);
  piPun_pr  ~ normal(0, 1.0); 
  rhoRew_pr ~ normal(0, 1.0);
  rhoPun_pr ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[4] wv_g;  # action wegith for go
    vector[4] wv_ng; # action wegith for nogo
    vector[4] qv_g;  # Q value for go
    vector[4] qv_ng; # Q value for nogo
    vector[4] sv;    # stimulus value 
    vector[4] pGo;   # prob of go (press) 

    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;
    sv    = initV;
  
    for (t in 1:Tsubj[i])  {
      if (sv[ cue[i,t] ] >= 0) {
        wv_g[ cue[i,t] ]  = qv_g[ cue[i,t] ] + b[i] + piRew[i] * sv[ cue[i,t] ];
      } else {
        wv_g[ cue[i,t] ]  = qv_g[ cue[i,t] ] + b[i] + piPun[i] * sv[ cue[i,t] ];
      }
      wv_ng[ cue[i,t] ] = qv_ng[ cue[i,t] ];  # qv_ng is always equal to wv_ng (regardless of action)      
      pGo[ cue[i,t] ]   = inv_logit( wv_g[ cue[i,t] ] - wv_ng[ cue[i,t] ] ); 
      pGo[ cue[i,t] ]   = pGo[ cue[i,t] ] * (1 - xi[i]) + xi[i]/2;  # noise
      pressed[i,t] ~ bernoulli( pGo[ cue[i,t] ] );
      
      # after receiving feedback, update sv[t+1]
      if (outcome[i,t] >= 0) {
        sv[ cue[i,t] ] = sv[ cue[i,t] ] + epRew[i] * ( rhoRew[i] * outcome[i,t] - sv[ cue[i,t] ] );
      } else {
        sv[ cue[i,t] ] = sv[ cue[i,t] ] + epPun[i] * ( rhoPun[i] * outcome[i,t] - sv[ cue[i,t] ] );
      }       

      # update action values
      if (pressed[i,t]) { # update go value 
        if (outcome[i,t] >=0) {
          qv_g[ cue[i,t] ] = qv_g[ cue[i,t] ] + epRew[i] * ( rhoRew[i] * outcome[i,t] - qv_g[ cue[i,t] ]);
        } else {
          qv_g[ cue[i,t] ] = qv_g[ cue[i,t] ] + epPun[i] * ( rhoPun[i] * outcome[i,t] - qv_g[ cue[i,t] ]);
        }
      } else { # update no-go value  
        if (outcome[i,t] >=0) {
          qv_ng[ cue[i,t] ] = qv_ng[ cue[i,t] ] + epRew[i] * ( rhoRew[i] * outcome[i,t] - qv_ng[ cue[i,t] ]);  
        } else{
          qv_ng[ cue[i,t] ] = qv_ng[ cue[i,t] ] + epPun[i] * ( rhoPun[i] * outcome[i,t] - qv_ng[ cue[i,t] ]);  
        }
      }  
    } # end of t loop
  } # end of i loop
}
generated quantities {
  real<lower=0, upper=1> mu_xi;
  real<lower=0, upper=1> mu_epRew;
  real<lower=0, upper=1> mu_epPun;
  real mu_b; 
  real<lower=0> mu_piRew;
  real<lower=0> mu_piPun;
  real<lower=0> mu_rhoRew;
  real<lower=0> mu_rhoPun;
  real log_lik[N];
  
  mu_xi     = Phi_approx(mu_p[1]);
  mu_epRew  = Phi_approx(mu_p[2]);
  mu_epPun  = Phi_approx(mu_p[3]);
  mu_b      = mu_p[4];
  mu_piRew  = exp(mu_p[5]);
  mu_piPun  = exp(mu_p[6]);
  mu_rhoRew = exp(mu_p[7]); 
  mu_rhoPun = exp(mu_p[8]); 
  
  { # local section, this saves time and space
    for (i in 1:N) {
      vector[4] wv_g;  # action wegith for go
      vector[4] wv_ng; # action wegith for nogo
      vector[4] qv_g;  # Q value for go
      vector[4] qv_ng; # Q value for nogo
      vector[4] sv;    # stimulus value 
      vector[4] pGo;   # prob of go (press) 
  
      wv_g  = initV;
      wv_ng = initV;
      qv_g  = initV;
      qv_ng = initV;
      sv    = initV;
    
      log_lik[i] = 0;

      for (t in 1:Tsubj[i])  {
        if (sv[ cue[i,t] ] >= 0) {
          wv_g[ cue[i,t] ]  = qv_g[ cue[i,t] ] + b[i] + piRew[i] * sv[ cue[i,t] ];
        } else {
          wv_g[ cue[i,t] ]  = qv_g[ cue[i,t] ] + b[i] + piPun[i] * sv[ cue[i,t] ];
        }
        wv_ng[ cue[i,t] ] = qv_ng[ cue[i,t] ];  # qv_ng is always equal to wv_ng (regardless of action)      
        pGo[ cue[i,t] ]   = inv_logit( wv_g[ cue[i,t] ] - wv_ng[ cue[i,t] ] ); 
        pGo[ cue[i,t] ]   = pGo[ cue[i,t] ] * (1 - xi[i]) + xi[i]/2;  # noise
        log_lik[i] = log_lik[i] + bernoulli_lpmf( pressed[i,t] | pGo[ cue[i,t] ] );
        
        # after receiving feedback, update sv[t+1]
        if (outcome[i,t] >= 0) {
          sv[ cue[i,t] ] = sv[ cue[i,t] ] + epRew[i] * ( rhoRew[i] * outcome[i,t] - sv[ cue[i,t] ] );
        } else {
          sv[ cue[i,t] ] = sv[ cue[i,t] ] + epPun[i] * ( rhoPun[i] * outcome[i,t] - sv[ cue[i,t] ] );
        }       
  
        # update action values
        if (pressed[i,t]) { # update go value 
          if (outcome[i,t] >=0) {
            qv_g[ cue[i,t] ] = qv_g[ cue[i,t] ] + epRew[i] * ( rhoRew[i] * outcome[i,t] - qv_g[ cue[i,t] ]);
          } else {
            qv_g[ cue[i,t] ] = qv_g[ cue[i,t] ] + epPun[i] * ( rhoPun[i] * outcome[i,t] - qv_g[ cue[i,t] ]);
          }
        } else { # update no-go value  
          if (outcome[i,t] >=0) {
            qv_ng[ cue[i,t] ] = qv_ng[ cue[i,t] ] + epRew[i] * ( rhoRew[i] * outcome[i,t] - qv_ng[ cue[i,t] ]);  
          } else{
            qv_ng[ cue[i,t] ] = qv_ng[ cue[i,t] ] + epPun[i] * ( rhoPun[i] * outcome[i,t] - qv_ng[ cue[i,t] ]);  
          }
        }  
      } # end of t loop
    } # end of i loop
  } # end of local section
}
