data {
  // number of observations
  int n;
  // number of variable == 1
  int p;
  // number of states
  int m;
  // size of system error
  int r;
  // observed data
  vector[p] y[n];
  // observation eq
  matrix[p, m] Z;
  // system eq
  matrix[m, m] T;
  matrix[m, r] R;
  // initial values
  vector[m] a_1;
  cov_matrix[m] P_1;
  // measurement error
  // vector<lower=0.0>[p] h;
  // vector<lower=0.0>[r] q; 
}
parameters {
  vector<lower=0.0>[p] h;
  vector<lower=0.0>[r] q;
}
transformed parameters {
  matrix[p, p] H;
  matrix[r, r] Q;
  H <- diag_matrix(h);
  Q <- diag_matrix(q);
}
model {
  vector[m] a[n + 1];
  matrix[m, m] P[n + 1];
  real llik_obs[n];
  real llik;
  // 1st observation
  a[1] <- a_1;
  P[1] <- P_1;
  for (i in 1:n) {
    vector[p] v;
    matrix[p, p] F;
    matrix[p, p] Finv;
    matrix[m, p] K;
    matrix[m, m] L;
    v <- y[i] - Z * a[i];
    F <- Z * P[i] * Z' + H;
    Finv <- inverse(F);
    K <- T * P[i] * Z' * Finv;
    L <- T - K * Z;
    // // manual update of multivariate normal
    llik_obs[i] <- -0.5 * (p * log(2 * pi()) + log(determinant(F)) + v' * Finv * v);
    //llik_obs[i] <- multi_normal_log(y[i], Z * a[i], F);
    a[i + 1] <- T * a[i] + K * v;
    P[i + 1] <- T * P[i] * L' + R * Q * R';
  }
  llik <- sum(llik_obs);
  lp__ <- lp__ + llik;
}