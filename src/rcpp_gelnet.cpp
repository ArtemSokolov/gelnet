#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// Worker for rcpp_gelnet_lin_obj() that take pre-computed fits
// Not exported
double rcpp_gelnet_lin_obj_w( arma::vec w, arma::vec s, arma::vec z,
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

  // L1-norm penalty
  arma::vec w1 = abs(w);
  if( d.isNotNull() ) w1 = w1 % as<arma::vec>(d);
  double R1 = l1 * sum(w1);

  // L2-norm penalty
  arma::vec w2 = w;
  if( m.isNotNull() ) w2 = w2 - as<arma::vec>(m);
  arma::mat w2t = w2.t();
  if( P.isNotNull() ) w2t = w2t * as<arma::mat>(P);
  double R2 = l2 * arma::as_scalar(w2t * w2) / 2.0;

  return L + R1 + R2;
}

//' Linear regression objective function value
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
double rcpp_gelnet_lin_obj( arma::vec w, double b, arma::mat X,
			    arma::vec z, double l1, double l2,
			    Nullable<NumericVector> a = R_NilValue,
			    Nullable<NumericVector> d = R_NilValue,
			    Nullable<NumericMatrix> P = R_NilValue,
			    Nullable<NumericVector> m = R_NilValue )
{
  arma::vec s = (X*w + b);
  return rcpp_gelnet_lin_obj_w( w, s, z, l1, l2, a, d, P, m );
}

//' Logistic regression objective function value
// [[Rcpp::export]]
arma::vec rcpp_gelnet_logreg_obj( arma::vec w, double b, arma::mat X, arma::Col<int> y,
				  double l1, double l2, bool balanced,
				  Nullable<NumericVector> d = R_NilValue,
				  Nullable<NumericMatrix> P = R_NilValue,
				  Nullable<NumericVector> m = R_NilValue )
{
  arma::vec s = (X*w + b);
  arma::vec ls = exp(s);
  ls.for_each( [](double& val) {val = std::log1p(val);} );
  return ls;
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

// Optimizes the GELNET linear objective via coordinate descent
// [[Rcpp::export]]
List rcpp_gelnet_lin_opt( arma::mat X, arma::vec z, double l1, double l2,
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
  double fprev = rcpp_gelnet_lin_obj_w( w, s, z, l1, l2, a, d, P, m );
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
      f = rcpp_gelnet_lin_obj_w( w, s, z, l1, l2, a, d, P, m );
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

