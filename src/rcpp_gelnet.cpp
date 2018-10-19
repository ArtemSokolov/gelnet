#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

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
  // Loss
  arma::vec err = z - (X*w + b);
  err = err % err;
  if( a.isNotNull() ) {err = err % as<arma::vec>(a);}
  double L = mean(err) / 2.0;

  // L1-norm penalty
  arma::vec w1 = abs(w);
  if( d.isNotNull() ) {w1 = w1 % as<arma::vec>(d);}
  double R1 = l1 * sum(w1);

  // L2-norm penalty
  arma::vec w2 = w;
  if( m.isNotNull() ) {w2 = w2 - as<arma::vec>(m);}
  arma::mat w2t = w2.t();
  if( P.isNotNull() ) {w2t = w2t * as<arma::mat>(P);}
  double R2 = l2 * arma::as_scalar(w2t * w2) / 2.0;

  return L + R1 + R2;
}
