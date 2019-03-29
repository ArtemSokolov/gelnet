#define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR
#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// Computes the L1-norm penalty
// Not exported
double l1penalty( double l1, arma::vec w,
		  Nullable<NumericVector> d = R_NilValue )
{
  arma::vec w1 = abs(w);
  if( d.isNotNull() ) w1 = w1 % as<arma::vec>(d);
  return l1 * sum(w1);
}

// Computes the L2-norm penalty
// Not exported
double l2penalty( double l2, arma::vec w,
		  Nullable<NumericMatrix> P = R_NilValue,
		  Nullable<NumericVector> m = R_NilValue )
{
  arma::vec w2 = w;
  if( m.isNotNull() ) w2 = w2 - as<arma::vec>(m);
  arma::mat w2t = w2.t();
  if( P.isNotNull() ) w2t = w2t * as<arma::mat>(P);
  return l2 * arma::as_scalar(w2t * w2) / 2.0;
}

// Worker for gelnet_lin_obj() that take pre-computed fits
// Not exported
double gelnet_lin_obj_w( arma::vec w, arma::vec s, arma::vec z,
			 double l1, double l2,
			 Nullable<NumericVector> a = R_NilValue,
			 Nullable<NumericVector> d = R_NilValue,
			 Nullable<NumericMatrix> P = R_NilValue,
			 Nullable<NumericVector> m = R_NilValue )
{
  // Loss
  arma::vec err = z - s;
  err = err % err;
  if( a.isNotNull() ) err = err % as<arma::vec>(a);
  double L = mean(err) / 2.0;

  // L1-norm and L2-norm penalties
  double R1 = l1penalty( l1, w, d );
  double R2 = l2penalty( l2, w, P, m );

  return L + R1 + R2;
}

// Worker for gelnet_blr_obj() that take pre-computed fits
// Not exported
double gelnet_blr_obj_w( arma::vec w, arma::vec s, arma::Col<int> y,
				  double l1, double l2, bool balanced,
				  Nullable<NumericVector> d = R_NilValue,
				  Nullable<NumericMatrix> P = R_NilValue,
				  Nullable<NumericVector> m = R_NilValue )
{
  // Compute fits and inverse labels
  arma::Col<int> y1 = 1 - y;

  // Compute logit transforms for individual samples
  arma::vec ls = exp(s);
  for( arma::uword i = 0; i < ls.n_elem; ++i )
    ls(i) = std::log1p(ls(i));

  // Per-class loss terms
  arma::vec lossp = y % (ls - s);
  arma::vec lossn = y1 % ls;

  // Overall loss term
  double L = 0.0;
  if( balanced )
    L = 0.5 * ( sum(lossp) / sum(y) + sum(lossn) / sum(y1) );
  else
    L = (sum(lossp) + sum(lossn)) / y.n_elem;

  // Regularization terms
  double R1 = l1penalty( l1, w, d );
  double R2 = l2penalty( l2, w, P, m );

  return L+R1+R2;
}

//' One-class logistic regression objective function
//'
//' Evaluates the one-class objective function value for a given model
//' See details.
//'
//' Computes the objective function value according to
//' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
//'  where
//' \deqn{ s_i = w^T x_i }
//' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
//'
//' @param w p-by-1 vector of model weights
//' @param X n-by-p matrix of n samples in p dimensions
//' @param l1 L1-norm penalty scaling factor \eqn{\lambda_1}
//' @param l2 L2-norm penalty scaling factor \eqn{\lambda_2}
//' @param d p-by-1 vector of feature weights
//' @param P p-by-p feature-feature penalty matrix
//' @param m p-by-1 vector of translation coefficients
//' @return The objective function value.
//' @seealso \code{\link{gelnet}}
//' @export
// [[Rcpp::export]]
double gelnet_oclr_obj( arma::vec w, arma::mat X, double l1, double l2,
		       Nullable<NumericVector> d = R_NilValue,
		       Nullable<NumericMatrix> P = R_NilValue,
		       Nullable<NumericVector> m = R_NilValue )
{
  // Compute the loss term
  arma::vec s = X*w;
  arma::vec ls = exp(s);
  for( arma::uword i = 0; i < ls.n_elem; ++i )
    ls(i) = std::log1p(ls(i));
  double L = -mean( s - ls );

  // Regularization terms
  double R1 = l1penalty( l1, w, d );
  double R2 = l2penalty( l2, w, P, m );

  return L+R1+R2;
}


//' Linear regression objective function
//'
//' Evaluates the linear regression objective function value for a given model.
//' See details.
//'
//' Computes the objective function value according to
//' \deqn{ \frac{1}{2n} \sum_i a_i (z_i - (w^T x_i + b))^2 + R(w) }
//'  where
//' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
//'
//' @param w p-by-1 vector of model weights
//' @param b the model bias term
//' @param X n-by-p matrix of n samples in p dimensions
//' @param z n-by-1 response vector
//' @param l1 L1-norm penalty scaling factor \eqn{\lambda_1}
//' @param l2 L2-norm penalty scaling factor \eqn{\lambda_2}
//' @param a n-by-1 vector of sample weights
//' @param d p-by-1 vector of feature weights
//' @param P p-by-p feature-feature penalty matrix
//' @param m p-by-1 vector of translation coefficients
//' @return The objective function value.
//' @export
// [[Rcpp::export]]
double gelnet_lin_obj( arma::vec w, double b, arma::mat X,
		       arma::vec z, double l1, double l2,
		       Nullable<NumericVector> a = R_NilValue,
		       Nullable<NumericVector> d = R_NilValue,
		       Nullable<NumericMatrix> P = R_NilValue,
		       Nullable<NumericVector> m = R_NilValue )
{
  arma::vec s = (X*w + b);
  return gelnet_lin_obj_w( w, s, z, l1, l2, a, d, P, m );
}

//' Binary logistic regression objective function
//'
//' Evaluates the logistic regression objective function value for a given model.
//' See details.
//'
//' Computes the objective function value according to
//' \deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
//'  where
//' \deqn{ s_i = w^T x_i + b }
//' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
//' When balanced is TRUE, the loss average over the entire data is replaced with averaging
//' over each class separately. The total loss is then computes as the mean over those
//' per-class estimates.
//'
//' @param w p-by-1 vector of model weights
//' @param b the model bias term
//' @param X n-by-p matrix of n samples in p dimensions
//' @param y n-by-1 binary response vector sampled from {0,1}
//' @param l1 L1-norm penalty scaling factor \eqn{\lambda_1}
//' @param l2 L2-norm penalty scaling factor \eqn{\lambda_2}
//' @param d p-by-1 vector of feature weights
//' @param P p-by-p feature-feature penalty matrix
//' @param m p-by-1 vector of translation coefficients
//' @param balanced boolean specifying whether the balanced model is being evaluated
//' @return The objective function value.
//' @seealso \code{\link{gelnet}}
//' @export
// [[Rcpp::export]]
double gelnet_blr_obj( arma::vec w, double b, arma::mat X, arma::Col<int> y,
			  double l1, double l2, bool balanced = false,
			  Nullable<NumericVector> d = R_NilValue,
			  Nullable<NumericMatrix> P = R_NilValue,
			  Nullable<NumericVector> m = R_NilValue )
{
  arma::vec s = (X*w + b);
  return gelnet_blr_obj_w( w, s, y, l1, l2, balanced, d, P, m );
}

// Soft threshold
// Not exported
double sthresh( double x, double a )
{
  if( fabs(x) < a ) return 0;
  else if( x < 0 ) return (x + a);
  else return (x - a);
}

// Computes the new value for coordinate j
// Not exported
double computeCoord( arma::mat X, arma::vec z, double l1, double l2,
		     arma::vec s, arma::vec Pwm, arma::vec w,
		     int j, bool nonneg,
		     Nullable<NumericVector> a = R_NilValue,
		     Nullable<NumericVector> d = R_NilValue,
		     Nullable<NumericMatrix> P = R_NilValue )
{
  // Retrieve dimensionality
  double n = X.n_rows;

  // Set up the adjustment value
  double adj = 1.0;
  if( P.isNotNull() ) adj = as<arma::mat>(P)(j,j);

  // Scale j^th feature by the sample weights
  arma::vec aX = X.col(j);
  if( a.isNotNull() ) aX = aX % as<arma::vec>(a);

  // Compute the numerator terms
  arma::vec Xj = X.col(j) * w(j);
  arma::vec err_j = z - s + Xj;
  double num1 = arma::as_scalar(aX.t() * err_j) / n;
  double num2 = l2 * (Pwm(j) - w(j)*adj);

  // Finalize the numerator
  double thresh = l1;
  if( d.isNotNull() ) thresh *= as<arma::vec>(d)(j);
  double num = sthresh( num1 - num2, thresh );

  // Early stopping
  if( num == 0.0 ) return 0.0;

  // Compute the denominator
  double denom = arma::as_scalar(aX.t() * X.col(j)) / n;
  denom += l2 * adj;

  // Compute the final results and check non-negativity constraints
  double res = num / denom;
  if( nonneg && res < 0.0 ) res = 0.0;
  return res;
}

//' GELnet optimizer for linear regression
//'
//' Constructs a GELnet model for linear regression using coordinate descent.
//'
//' The method operates through cyclical coordinate descent.
//' The optimization is terminated after the desired tolerance is achieved, or after a maximum number of iterations.
//'
//' @param X n-by-p matrix of n samples in p dimensions
//' @param z n-by-1 vector of response values
//' @param l1 coefficient for the L1-norm penalty
//' @param l2 coefficient for the L2-norm penalty
//' @param a n-by-1 vector of sample weights
//' @param d p-by-1 vector of feature weights
//' @param P p-by-p feature association penalty matrix
//' @param m p-by-1 vector of translation coefficients
//' @param max_iter maximum number of iterations
//' @param eps convergence precision
//' @param w_init initial parameter estimate for the weights
//' @param b_init initial parameter estimate for the bias term
//' @param fix_bias set to TRUE to prevent the bias term from being updated (default: FALSE)
//' @param silent set to TRUE to suppress run-time output; overwrites verbose (default: FALSE)
//' @param verbose set to TRUE to see extra output; is overwritten by silent (default: FALSE)
//' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
//' @return A list with two elements:
//' \describe{
//'   \item{w}{p-by-1 vector of p model weights}
//'   \item{b}{scalar, bias term for the linear model}
//' }
//' @export
// [[Rcpp::export]]
List gelnet_lin_opt( arma::mat X, arma::vec z, double l1, double l2,
		     int max_iter = 100, double eps = 1e-5, bool fix_bias = false,
		     bool silent = false, bool verbose = false, bool nonneg = false,
		     Nullable<NumericVector> w_init = R_NilValue,
		     Nullable<double> b_init = R_NilValue,
		     Nullable<NumericVector> a = R_NilValue,
		     Nullable<NumericVector> d = R_NilValue,
		     Nullable<NumericMatrix> P = R_NilValue,
		     Nullable<NumericVector> m = R_NilValue )
{
  // Retrieve data dimensionality
  int n = X.n_rows;
  int p = X.n_cols;

  // Initialize the model (using provided values if available)
  arma::vec w; double b;
  if( w_init.isNotNull() ) w = as<arma::vec>(w_init);
  else w.zeros(p);
  if( b_init.isNotNull() ) b = as<double>(b_init);
  else {
    arma::vec num = z; double denom = n;
    if( a.isNotNull() ) {
      arma::vec a_ = as<arma::vec>(a);
      num = num % a_; denom = sum(a_);
    }
    b = sum(num) / denom;
  }

  // Compute the initial fits and objective function value
  arma::vec s = (X*w + b);
  double fprev = gelnet_lin_obj_w( w, s, z, l1, l2, a, d, P, m );
  if( !silent )
    Rcout<<"Initial objective value: "<<fprev<<std::endl;

  // Cache the P(w-m) term
  arma::vec Pwm = w;
  if( m.isNotNull() ) Pwm = Pwm - as<arma::vec>(m);
  if( P.isNotNull() ) Pwm = as<arma::mat>(P) * Pwm;

  // Perform coordinate descent
  int iter; double f;
  for( iter = 1; iter <= max_iter; ++iter )
    {
      Rcpp::checkUserInterrupt();

      // Update the weights
      for( int j = 0; j < p; ++j )
	{
	  // Perform the update and compute the difference
	  double wj_old = w(j);
	  w(j) = computeCoord( X, z, l1, l2, s, Pwm, w, j, nonneg, a, d, P );
	  double wj_diff = w(j) - wj_old;

	  // Update fits and L2-norm penalty term accordingly
	  if( wj_diff != 0.0 )
	    {
	      s = s + X.col(j) * wj_diff;
	      if( P.isNotNull() ) Pwm = Pwm + as<arma::mat>(P).col(j) * wj_diff;
	      else Pwm(j) = Pwm(j) + wj_diff;
	    }
	}

      // Update the bias term
      if( !fix_bias )
	{
	  // Compute the numerator and denominator
	  arma::vec err = z - s + b;
	  double b_denom = n;
	  if( a.isNotNull() )
	    {
	      arma::vec a_ = as<arma::vec>(a);
	      err = err % a_;
	      b_denom = sum(a_);
	    }
	  double b_num = sum( err );

	  // Update the term and compute the difference
	  double b_old = b;
	  b = b_num / b_denom;
	  double b_diff = b - b_old;

	  // Update fits accordingly
	  if( b_diff != 0.0 ) s = s + b_diff;
	}

      // Compute the new objective function value
      f = gelnet_lin_obj_w( w, s, z, l1, l2, a, d, P, m );
      if( !silent && verbose )
	Rcout<<"After iteration "<<iter<<": "<<f<<std::endl;
      if( fabs( f - fprev ) / fabs( fprev ) < eps ) break;
      else fprev = f;
    }
  if( iter > max_iter ) --iter;    // Corner case: loop didn't end via break
  if( !silent )
    Rcout<<"Final value is "<<f<<" after iteration "<<iter<<std::endl;

  return List::create( Named("w") = w, Named("b") = b );
}

//' GELnet optimizer for binary logistic regression
//'
//' Constructs a GELnet model for logistic regression using the Newton method.
//'
//' The method operates by constructing iteratively re-weighted least squares approximations
//' of the log-likelihood loss function and then calling the linear regression routine
//' to solve those approximations. The least squares approximations are obtained via the Taylor series
//' expansion about the current parameter estimates.
//'
//' @param X n-by-p matrix of n samples in p dimensions
//' @param y n-by-1 vector of binary response labels (must be in {0,1})
//' @param l1 coefficient for the L1-norm penalty
//' @param l2 coefficient for the L2-norm penalty
//' @param d p-by-1 vector of feature weights
//' @param P p-by-p feature association penalty matrix
//' @param m p-by-1 vector of translation coefficients
//' @param max_iter maximum number of iterations
//' @param eps convergence precision
//' @param w_init initial parameter estimate for the weights
//' @param b_init initial parameter estimate for the bias term
//' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
//' @param balanced boolean specifying whether the balanced model is being trained
//' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
//' @return A list with two elements:
//' \describe{
//'   \item{w}{p-by-1 vector of p model weights}
//'   \item{b}{scalar, bias term for the linear model}
//' }
//' @seealso \code{\link{gelnet.lin}}
//' @export
// [[Rcpp::export]]
List gelnet_blr_opt( arma::mat X, arma::Col<int> y, double l1, double l2,
			int max_iter = 100, double eps = 1e-5,
			bool silent = false, bool verbose = false,
			bool balanced = false, bool nonneg = false,
			Nullable<NumericVector> w_init = R_NilValue,
			Nullable<double> b_init = R_NilValue,
			Nullable<NumericVector> d = R_NilValue,
			Nullable<NumericMatrix> P = R_NilValue,
			Nullable<NumericVector> m = R_NilValue )
{
  // Retrieve data dimensionality
  int n = X.n_rows;
  int p = X.n_cols;
  double npos = sum(y);
  double nneg = n - npos;

  // Initialize the model (using provided values if available)
  arma::vec w; double b;
  if( w_init.isNotNull() ) w = as<arma::vec>(w_init);
  else w.zeros(p);
  if( b_init.isNotNull() ) b = as<double>(b_init);
  else b = 0.0;

  // Compute the initial fits and objective function value
  arma::vec s = (X*w + b);
  double fprev = gelnet_blr_obj_w( w, s, y, l1, l2, balanced, d, P, m );
  if( !silent ) Rcout<<"Initial objective value: "<<fprev<<std::endl;

  // Main optimization loop
  int iter; double f = 0.0;
  for( iter = 1; iter <= max_iter; ++iter )
    {
      if( !silent && verbose )
	Rcout<<"Iteration "<<iter<<": "<<"f = "<<fprev<<std::endl;

      // Compute the current fit
      arma::vec pr = 1 / (1 + exp(-s));
      arma::vec a = pr % (1-pr);

      // Handle near-zero and near-one probability values
      for( arma::uword i = 0; i < pr.n_elem; ++i )
	{
	  if( pr(i) < eps ) { pr(i) = 0.0; a(i) = eps; }
	  else if( pr(i) > (1-eps) ) { pr(i) = 1.0; a(i) = eps; }
	}

      // Compute the response
      arma::vec z = s + (y-pr) / a;

      // Rebalance the sample weights according to the class counts
      if( balanced )
	{
	  for( arma::uword i = 0; i < y.n_elem; ++i )
	    {
	      if( y(i) == 0 ) a(i) *= n / (nneg*2);
	      else a(i) *= n / (npos*2);
	    }
	}

      // Perform coordinate descent
      int nIter = iter * 2;
      NumericVector w0 = wrap(w);
      NumericVector a0 = wrap(a);
      Nullable<double> b0 = wrap(b);
      List newModel = gelnet_lin_opt( X, z, l1, l2, nIter, eps, false,
				      true, false, nonneg, w0, b0, a0, d, P, m );

      // Recompute the fits
      w = as<arma::vec>( newModel["w"] );
      b = newModel["b"];
      s = (X*w + b);

      // Compute the objective function value and check the stopping criterion
      f = gelnet_blr_obj_w( w, s, y, l1, l2, balanced, d, P, m );
      if( fabs( f - fprev ) / fabs( fprev ) < eps ) break;
      else fprev = f;
    }

  if( iter > max_iter ) --iter;    // Corner case: loop didn't end via break
  if( !silent )
    Rcout<<"Final value is "<<f<<" after iteration "<<iter<<std::endl;

  return List::create( Named("w") = w, Named("b") = b );
}

//' GELnet optimizer for one-class logistic regression
//'
//' Constructs a GELnet model for one-class regression using the Newton method.
//'
//' The function optimizes the following objective:
//' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
//'  where
//' \deqn{ s_i = w^T x_i }
//' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
//' The method operates by constructing iteratively re-weighted least squares approximations
//' of the log-likelihood loss function and then calling the linear regression routine
//' to solve those approximations. The least squares approximations are obtained via the Taylor series
//' expansion about the current parameter estimates.
//'
//' @param X n-by-p matrix of n samples in p dimensions
//' @param l1 coefficient for the L1-norm penalty
//' @param l2 coefficient for the L2-norm penalty
//' @param d p-by-1 vector of feature weights
//' @param P p-by-p feature association penalty matrix
//' @param m p-by-1 vector of translation coefficients
//' @param max_iter maximum number of iterations
//' @param eps convergence precision
//' @param w_init initial parameter estimate for the weights
//' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
//' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
//' @return A list with one element:
//' \describe{
//'   \item{w}{p-by-1 vector of p model weights}
//' }
//' @export
// [[Rcpp::export]]
List gelnet_oclr_opt( arma::mat X, double l1, double l2, int max_iter = 100,
		      double eps = 1e-5, bool silent = false, bool verbose = false,
		      bool nonneg = false,
		      Nullable<NumericVector> w_init = R_NilValue,
		      Nullable<NumericVector> d = R_NilValue,
		      Nullable<NumericMatrix> P = R_NilValue,
		      Nullable<NumericVector> m = R_NilValue )
{
  // Retrieve data dimensionality
  int p = X.n_cols;

  // Initialize the model (using provided values if available)
  arma::vec w;
  if( w_init.isNotNull() ) w = as<arma::vec>(w_init);
  else w.zeros(p);

  // Compute the initial fits and objective function value
  double fprev = gelnet_oclr_obj( w, X, l1, l2, d, P, m );
  if( !silent ) Rcout<<"Initial objective value: "<<fprev<<std::endl;

  // Main optimization loop
  int iter; double f = 0.0;
  for( iter = 1; iter <= max_iter; ++iter )
    {
      if( !silent && verbose )
	Rcout<<"Iteration "<<iter<<": "<<"f = "<<fprev<<std::endl;

      // Compute the current fit
      arma::vec s = X*w;
      arma::vec pr = 1 / (1 + exp(-s));
      arma::vec a = pr % (1-pr);
      arma::vec z = s + 1/pr;

      // Perform coordinate descent
      int nIter = iter * 2;
      NumericVector w0 = wrap(w);
      NumericVector a0 = wrap(a);
      Nullable<double> b0 = wrap(0.0);
      List newModel = gelnet_lin_opt( X, z, l1, l2, nIter, eps, true,
				      true, false, nonneg, w0, b0, a0, d, P, m );

      // Retrieve the model
      w = as<arma::vec>( newModel["w"] );

      // Compute the objective function value and check the stopping criterion
      f = gelnet_oclr_obj( w, X, l1, l2, d, P, m );
      if( fabs( f - fprev ) / fabs( fprev ) < eps ) break;
      else fprev = f;
    }

  return List::create( Named("w") = w );
}
