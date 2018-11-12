#include <R.h>
#include <math.h>

// Soft threshold
void sthresh( double* x, double a )
{
  if( fabs(*x) < a ) *x = 0;
  else if( *x < 0 ) *x = *x + a;
  else *x = *x - a;
}

// Updates fits and L2-norm penalties
void updateFits( double* X, double* K, double* S, double* Kwm,
		 int* np, int* pp, int* jp, double* wj_diffp )
{
  int n = *np;
  int p = *pp;
  int j = *jp;
  double wjd = *wj_diffp;

  for( int i = 0; i < n; ++i )
    S[i] += X[i+j*n] * wjd;

  for( int k = 0; k < p; ++k )
    Kwm[k] += K[k+j*p] * wjd;
}

// Computes the GELNET objective function value
void gelnet_lin_obj( double* S, double* z, double* a, double* d, double* Kwm,
		     double* m, double* w, double* lambda1p,
		     double* lambda2p, int* np, int* pp, double* res )
{
  int n = *np;
  int p = *pp;
  double lambda1 = *lambda1p;
  double lambda2 = *lambda2p;

  // Loss term
  double loss = 0.0;
  for( int i = 0; i < n; ++i )
    {
      double r = (z[i] - S[i]);
      loss += a[i] * r * r;
    }

  // Regularization terms
  double regL1 = 0.0;
  double regL2 = 0.0;
  for( int j = 0; j < p; ++j )
    {
      regL1 += d[j] * fabs( w[j] );
      regL2 += (w[j] - m[j]) * Kwm[j];
    }

  *res = 0.5 * loss / n + lambda1 * regL1 + 0.5 * lambda2 * regL2;
}

// Computes the GELNET logistic regression objection function value
void gelnet_logreg_obj( double* S, int* y, double* d, double* Kwm, double* m,
			double* w, double* lambda1p, double* lambda2p, 
			int* np, int* pp, double* res, int* bBalanced )
{
  int n = *np;
  int p = *pp;
  double lambda1 = *lambda1p;
  double lambda2 = *lambda2p;

  // Loss term on a per-class basis
  double lossp = 0.0;
  double lossn = 0.0;
  int nneg = 0;
  for( int i = 0; i < n; ++i )
    {
      if( y[i] == 0 )
	{ lossn += log1p( exp(S[i]) ); nneg++; }
      else
	lossp += log1p( exp(S[i]) ) - y[i]*S[i];
    }

  // Overall loss term
  double loss = 0.0;
  if( *bBalanced )
    loss = 0.5 * (lossp / (n - nneg) + lossn / nneg);
  else
    loss = (lossp + lossn) / n;

  // Regularization terms
  double regL1 = 0.0;
  double regL2 = 0.0;
  for( int j = 0; j < p; ++j )
    {
      regL1 += d[j] * fabs( w[j] );
      regL2 += (w[j] - m[j]) * Kwm[j];
    }

  *res = loss + lambda1 * regL1 + 0.5 * lambda2 * regL2;
}

// Computes the new value for coordinate *jp
void computeCoord( double* X, double* z, double* a, double* d, double* K,
		   double* lambda1p, double* lambda2p, double* S,
		   double* Kwm, int* np, int* pp, int* jp, double* w, 
		   double* work_a1, double* res, int* bNonneg )
{
  // Dereference
  int n = *np; int p = *pp; int j = *jp;
  double lambda1 = *lambda1p;
  double lambda2 = *lambda2p;

  // Compute the working space values
  for( int i = 0; i < n; ++i )
    work_a1[i] = a[i] * X[i+j*n];

  // Compute the numerator
  double num = 0.0;
  for( int i = 0; i < n; ++i )
    num += work_a1[i] * (z[i] - S[i] + X[i+j*n] * w[j]);

  // Normalize the numerator
  num /= n;
  num -= lambda2 * (Kwm[j] - K[j+j*p] * w[j]);
  sthresh( &num, lambda1*d[j] );

  // Early stopping
  if( num == 0.0 ) { *res = 0.0; return; }

  // Compute the denominator
  double denom = 0.0;
  for( int i = 0; i < n; ++i )
    denom += work_a1[i] * X[i+j*n];

  // Normalize the denominator
  denom /= n;
  denom += lambda2 * K[j+j*p];

  *res = num / denom;

  // Check the non-negativity constraint
  if( *bNonneg && *res < 0.0 )
    *res = 0.0;
}

// Optimizes the GELNET objective via coordinate descent
void gelnet_lin_opt( double* X, double* z, double* a, double* d, double* K, 
		      double* m, double* lambda1p, double* lambda2p,
		      double* S, double* Kwm, int* np, int* pp,
		      int* max_iter, double* eps, int* fix_bias,
		     double* w, double* b, int* bSilentp, int* bNonneg )
{
  // Dereference
  int n = *np; int p = *pp;

  // Working storage
  double* work_a1 = (double*) R_alloc( n, sizeof( double ) );

  if( !(*bSilentp) )
    Rprintf( "Running linear regression optimization with L1 = %f, L2 = %f\n",
	     *lambda1p, *lambda2p );

  // Compute the initial objective function value
  double fprev;
  gelnet_lin_obj( S, z, a, d, Kwm, m, w, lambda1p, lambda2p, np, pp, &fprev );

  // Perform coordinate descent
  int iter; double f;
  for( iter = 1; iter <= (*max_iter); ++iter )
    {
      // Update the weights
      for( int j = 0; j < p; ++j )
	{
	  // Perform the update
	  double wj_old = w[j];
	  computeCoord( X, z, a, d, K, lambda1p, lambda2p, S, Kwm,
			np, pp, &j, w, work_a1, w+j, bNonneg );

	  // Update fits and L2-norm penalty term accordingly
	  double wj_diff = w[j] - wj_old;
	  if( wj_diff != 0.0 )
	    updateFits( X, K, S, Kwm, np, pp, &j, &wj_diff);
	}

      // Update the bias term
      if( !(*fix_bias) )
	{
	  double b_num = 0.0;
	  double b_denom = 0.0;
	  for( int i = 0; i < n; ++i )
	    {
	      double s = S[i] - *b;
	      b_num += a[i] * (z[i] - s);
	      b_denom += a[i];
	    }
      
	  double b_old = *b;
	  *b = b_num / b_denom;
	  double b_diff = *b - b_old;

	  // Update the fits accordingly
	  if( b_diff != 0 )
	    {
	      for( int i = 0; i < n; ++i )
		S[i] += b_diff;
	    }
	}

      // Compute the objective function value and check the stopping criterion
      gelnet_lin_obj( S, z, a, d, Kwm, m, w, lambda1p, lambda2p, np, pp, &f );
      if( fabs( f - fprev ) / fabs( fprev ) < *eps ) break;
      else fprev = f;
    }
  if( iter > (*max_iter) ) --iter;	// Corner case: loop didn't end via break
  if( !(*bSilentp) )
    Rprintf( "f = %f after iteration %d\n", f, iter );
}

// Optimizes the GELNET logistic regression objective
void gelnet_logreg_opt( double* X, int* y, double* d, double* K, double* m,
			double* lambda1p, double* lambda2p,
			double* S, double* Kwm, int* np, int* pp,
			int* max_iter, double* eps, double* w, double* b,
			int* bSilentp, int* bBalanced, int* bNonneg )
{
  int n = *np; int p = *pp;
  double lambda1 = *lambda1p;
  double lambda2 = *lambda2p;

  // Working storage
  double* a = (double*) R_alloc( n, sizeof( double ) );
  double* z = (double*) R_alloc( n, sizeof( double ) );

  if( !(*bSilentp) )
    Rprintf( "Running logistic regression optimization with L1 = %f, L2 = %f\n",
	     *lambda1p, *lambda2p );

  // Compute the initial objective function value
  double fprev;
  gelnet_logreg_obj( S, y, d, Kwm, m, w, lambda1p, lambda2p, np, pp, &fprev, bBalanced );

  // Count the number of samples in each class
  double npos = 0.0;
  double nneg = 0.0;
  for( int i = 0; i < n; ++i )
    {
      if( y[i] == 0 ) nneg += 1.0;
      else npos += 1.0;
    }

  // Run the optimization
  int iter; double f;
  for( iter = 1; iter <= (*max_iter); ++iter )
    {
      if( !(*bSilentp) )
	{
	  Rprintf( "Iteration %d: ", iter );
	  Rprintf( "f = %f\n", fprev );
	}

      // Compute the current fit
      for( int i = 0; i < n; ++i )
	{
	  // Compute the sample weight
	  double pr = 1 / (1 + exp( -S[i] ));
	  if( pr < *eps ) { pr = 0.0; a[i] = *eps; }
	  else if( pr > (1.0-*eps) ) { pr = 1.0; a[i] = *eps; }
	  else a[i] = pr * (1-pr);

	  // Compute the response
	  z[i] = S[i] + (y[i] - pr) / a[i];
	}

      // Rebalance the sample weights according to the class counts
      if( *bBalanced )
	{
	  for( int i = 0; i < n; ++i )
	    {
	      if( y[i] == 0 )
		a[i] *= n / (nneg*2);
	      else
		a[i] *= n / (npos*2);
	    }
	}

      // Perform coordinate descent
      int nIter = iter * 2;
      int bSilent = 1;
      int fix_bias = 0;
      gelnet_lin_opt( X, z, a, d, K, m, lambda1p, lambda2p, S, Kwm, np, pp,
		      &nIter, eps, &fix_bias, w, b, &bSilent, bNonneg );

      // Compute the objective function value and check the stopping criterion
      gelnet_logreg_obj( S, y, d, Kwm, m, w, lambda1p, lambda2p, np, pp, &f, bBalanced );
      if( fabs( f - fprev ) / fabs( fprev ) < *eps ) break;
      else fprev = f;
    }
}
