## Generalized Elastic Nets
##
## by Artem Sokolov

#' The largest meaningful value of the L1 parameter
#'
#' Computes the smallest value of the LASSO coefficient L1 that leads to an
#'  all-zero weight vector for a given linear regression problem.
#'
#' The cyclic coordinate descent updates the model weight \eqn{w_k} using a soft threshold operator
#' \eqn{ S( \cdot, \lambda_1 d_k ) } that clips the value of the weight to zero, whenever the absolute
#' value of the first argument falls below \eqn{\lambda_1 d_k}. From here, it is straightforward to compute
#' the smallest value of \eqn{\lambda_1}, such that all weights are clipped to zero.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of response values. Must be numeric vector for regression, factor with 2 levels for binary classification, or NULL for a one-class task.
#' @param a n-by-1 vector of sample weights (regression only)
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param l2 coefficient for the L2-norm penalty
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @return The largest meaningful value of the L1 parameter (i.e., the smallest value that yields a model with all zero weights)
#' @export
L1.ceiling <- function( X, y, a = rep(1,nrow(X)), d = rep(1,ncol(X)),
                       P = diag(ncol(X)), m = rep(0,ncol(X)), l2 = 1, balanced = FALSE )
  {
    if( is.null(y) )
      {
        ## One-class model
        xy <- (apply( X/2, 2, mean ) + l2 * P %*% m) / d
        return( max( abs(xy) ) )
      }
    else if( is.factor(y) )
      {
        ## Binary model
        ## Convert the labels to {0,1}
        y <- as.integer( y == levels(y)[1] )

        a <- rep( 0.25, nrow(X) )
        if( balanced )
          {
            jpos <- which( y==1 ); jneg <- which( y==0 )
            a[jpos] <- a[jpos] * length(y) / ( length(jpos)*2 )
            a[jneg] <- a[jneg] * length(y) / ( length(jneg)*2 )
          }
        z <- (y - 0.5) / 0.25
        return( L1.ceiling( X, z, a, d, P, m, l2 ) )
      }
    else if( is.numeric(y) )
      {
        ## Liner regression model
        stopifnot( nrow(X) == length(y) )
        b1 <- sum( a*y ) / sum(a)
        xy <- (apply( a*X*(y - b1), 2, mean ) + l2 * P %*% m) / d
        return( max( abs(xy) ) )
      }
    else
      { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }
  }

#' GELnet for linear regression, binary classification and one-class problems.
#'
#' Infers the problem type and learns the appropriate GELnet model via coordinate descent.
#'
#' The method determines the problem type from the labels argument y.
#' If y is a numeric vector, then a regression model is trained by optimizing the following objective function:
#' \deqn{ \frac{1}{2n} \sum_i a_i (y_i - (w^T x_i + b))^2 + R(w) }
#'
#' If y is a factor with two levels, then the function returns a binary classification model, obtained by optimizing the following objective function:
#' \deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i + b }
#'
#' Finally, if no labels are provided (y == NULL), then a one-class model is constructed using the following objective function:
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i }
#'
#' In all cases, the regularizer is defined by
#' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#'
#' The training itself is performed through cyclical coordinate descent, and the optimization is terminated after the desired tolerance is achieved or after a maximum number of iterations.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of response values. Must be numeric vector for regression, factor with 2 levels for binary classification, or NULL for a one-class task.
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param nFeats alternative parameterization that returns the desired number of non-zero weights. Takes precedence over l1 if not NULL (default: NULL)
#' @param a n-by-1 vector of sample weights (regression only)
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param b.init initial parameter estimate for the bias term
#' @param fix.bias set to TRUE to prevent the bias term from being updated (regression only) (default: FALSE)
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @return A list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model (omitted for one-class models)}
#' }
#' @export
gelnet <- function( X, y, l1, l2, nFeats=NULL, a=rep(1,n), d=rep(1,p), P=diag(p), m=rep(0,p),
                   max.iter=100, eps=1e-5, w.init=rep(0,p), b.init=NULL,
                   fix.bias=FALSE, silent=FALSE, balanced=FALSE, nonneg=FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## One-class
    if( is.null(y) )
    {
        if( !silent ) cat( "Training a one-class model\n" )
        f.gel <- function(L1)
        {gelnet.oneclass( X, L1, l2, d, P, m, max.iter, eps, w.init, silent, nonneg )}
    }

    ## Binary classification
    else if( is.factor(y) )
      {
          if( nlevels(y) == 1 )
              stop( "All labels are identical\nConsider training a one-class model instead" )
          if( nlevels(y) > 2 )
              stop( "Labels belong to a multiclass task\nConsider training a set of one-vs-one or one-vs-rest models" )
          if( !silent ) cat( "Training a logistic regression model\n" )
          if( is.null(b.init) ) b.init <- 0

          ## Convert the labels to {0,1}
          y <- factor(y)
          if( !silent )
              cat( "Treating", levels(y)[1], "as the positive class\n" )
          y <- as.integer( y == levels(y)[1] )

          f.gel <- function(L1)
          {gelnet.logreg( X, y, L1, l2, d, P, m, max.iter, eps,
                         w.init, b.init, silent, balanced, nonneg )}
      }

    ## Regression
    else if( is.numeric(y) )
      {
          if( !silent ) cat( "Training a linear regression model\n" )
          if( is.null(b.init) ) b.init <- sum(a*y) / sum(a)
          f.gel <- function(L1)
          {gelnet.lin( X, y, L1, l2, a, d, P, m, max.iter, eps,
                      w.init, b.init, fix.bias, silent, nonneg )}
      }
    else
      { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }

    ## Train a model with the required number of features (if requested)
    if( !is.null(nFeats) )
    {
        L1s <- L1.ceiling( X, y, a, d, P, m, l2, balanced )
        return( gelnet.L1bin( f.gel, nFeats, L1s ) )
    }
    else
      { return( f.gel(l1) ) }
  }

#' k-fold cross-validation for parameter tuning.
#'
#' Performs k-fold cross-validation to select the best pair of the L1- and L2-norm penalty values.
#'
#' Cross-validation is performed on a grid of parameter values. The user specifies the number of values
#' to consider for both the L1- and the L2-norm penalties. The L1 grid values are equally spaced on
#' [0, L1s], where L1s is the smallest meaningful value of the L1-norm penalty (i.e., where all the model
#' weights are just barely zero). The L2 grid values are on a logarithmic scale centered on 1.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of response values. Must be numeric vector for regression, factor with 2 levels for binary classification, or NULL for a one-class task.
#' @param nL1 number of values to consider for the L1-norm penalty
#' @param nL2 number of values to consider for the L2-norm penalty
#' @param nFolds number of cross-validation folds (default:5)
#' @param a n-by-1 vector of sample weights (regression only)
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param b.init initial parameter estimate for the bias term
#' @param fix.bias set to TRUE to prevent the bias term from being updated (regression only) (default: FALSE)
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @return A list with the following elements:
#' \describe{
#'   \item{l1}{the best value of the L1-norm penalty}
#'   \item{l2}{the best value of the L2-norm penalty}
#'   \item{w}{p-by-1 vector of p model weights associated with the best (l1,l2) pair.}
#'   \item{b}{scalar, bias term for the linear model associated with the best (l1,l2) pair. (omitted for one-class models)}
#'   \item{perf}{performance value associated with the best model. (Likelihood of data for one-class, AUC for binary classification, and -RMSE for regression)}
#' }
#' @seealso \code{\link{gelnet}}
#' @export
gelnet.cv <- function( X, y, nL1, nL2, nFolds=5, a=rep(1,n), d=rep(1,p), P=diag(p), m=rep(0,p),
                   max.iter=100, eps=1e-5, w.init=rep(0,p), b.init=0,
                   fix.bias=FALSE, silent=FALSE, balanced=FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## One-class
    if( is.null(y) )
      {
        ## Assign every k^th sample to its fold
        vfa <- rep( 1:nFolds, length=n )

        ## The evaluation function computes the likelihood of the test data given the model
        f.eval <- function( mm, X.te, y.te )
          {
            s <- drop( X.te %*% mm$w )
            pr <- exp(s) / (1 + exp(s))
            sum( log( pr ) )
          }
      }

    ## Binary classification
    else if( is.factor(y) )
      {
        if( nlevels(y) == 1 )
          stop( "All labels are identical\nConsider training a one-class model instead" )
        if( nlevels(y) > 2 )
          stop( "Labels belong to a multiclass task\nConsider training a set of one-vs-one or one-vs-rest models" )

        ## Sample each class separately
        vfa <- rep( 0, length=n )
        jpos <- which( y == levels(y)[1] )
        jneg <- which( y == levels(y)[2] )
        vfa[jpos] <- rep(1:nFolds, length.out = length(jpos) )
        vfa[jneg] <- rep(1:nFolds, length.out = length(jneg) )

        ## The evaluation function computes AUC on the test data/labels
        f.eval <- function( mm, X.te, y.te )
          {
            s <- drop( X.te %*% mm$w + mm$b )
            jp <- which( y.te == levels(y)[1] ); np <- length( jp )
            jn <- which( y.te == levels(y)[2] ); nn <- length( jn )
            s0 <- sum( rank(s)[jp] )
            (s0 - np*(np+1) / 2) / (np*nn)
          }
      }

    ## Regression
    else if( is.numeric(y) )
      {
        ## Assign every k^th sample to a fold to maintain data representation across folds
        vfa <- rep( 0, length=n )
        vfa[order(y)] <- rep( 1:nFolds, length=n )

        ## The evaluation function computes -RMSE on the test data/labels
        ## Negative RMSE is used for consistency with other "higher is better" measures
        f.eval <- function( mm, X.te, y.te )
          {
            s <- drop( X.te %*% mm$w + mm$b )
            -sqrt( mean( (s - y.te)^2 ) )
          }
      }
    else
      { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }

    ## Generate a vector of values for the L2-norm penalty
    v1 <- 1:(as.integer( nL2 / 2 ))
    if( nL2 %% 2 == 0 )
      vL2 <- 10 ^ c(rev(1-v1), v1)
    else
      vL2 <- 10 ^ c(rev(-v1), 0, v1)
    stopifnot( length(vL2) == nL2 )

    ## Traverse the L2-norm penalty values
    mm.best <- list( perf = -Inf )
    for( l2 in vL2 )
      {
        ## Generate a vector of values for the L1-norm penalty
        L1s <- L1.ceiling( X, y, a, d, P, m, l2, balanced )
        vL1 <- seq( 0, L1s, length.out = nL1+1 )[1:nL1]
        stopifnot( length(vL1) == nL1 )

        ## Traverse the L1-norm penalty values
        for( l1 in vL1 )
          {
            if( !silent )
              {
                cat( "===================================================\n" )
                cat( "Evaluating the choice of L2 =", l2, "; L1 =", l1, "\n" )
              }

            ## Traverse the folds
            res <- rep( 0, nFolds )
            for( k in 1:nFolds )
              {
                if( !silent )
                  cat( "== Fold", k, "==\n" )

                ## Isolate the training and test data
                j.te <- which( vfa == k )
                j.tr <- which( vfa != k )

                ## Train and evaluate the model
                mm <- gelnet( X[j.tr,], y[j.tr], l1, l2, NULL, a[j.tr], d, P, m,
                             max.iter, eps, w.init, b.init,
                             fix.bias, silent, balanced )
                res[k] <- f.eval( mm, X[j.te,], y[j.te] )

                if( !silent )
                cat( "Performance:", res[k], "\n" )
              }
            cat( "Average performance across all folds: ", mean(res), "\n" )

            ## Compare to and store the best model
            if( mean(res) > mm.best$perf )
              {
                mm.best <- mm
                mm.best$l1 <- l1
                mm.best$l2 <- l2
                mm.best$perf <- mean(res)
              }
          }
      }

    mm.best
  }

#' GELnet for linear regression
#'
#' Constructs a GELnet model for linear regression using coordinate descent.
#'
#' The method operates through cyclical coordinate descent.
#' The optimization is terminated after the desired tolerance is achieved, or after a maximum number of iterations.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param z n-by-1 vector of response values
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param a n-by-1 vector of sample weights
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param b.init initial parameter estimate for the bias term
#' @param fix.bias set to TRUE to prevent the bias term from being updated (default: FALSE)
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @return A list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model}
#' }
#' @noRd
gelnet.lin <- function( X, z, l1, l2, a = rep(1,n), d = rep(1,p), P = diag(p),
                       m=rep(0,p), max.iter = 100, eps = 1e-5, w.init = rep(0,p),
                       b.init = sum(a*z)/sum(a), fix.bias=FALSE, silent=FALSE, nonneg=FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Verify argument dimensionality
    stopifnot( length(z) == n )
    stopifnot( length(a) == n )
    stopifnot( length(d) == p )
    stopifnot( all( dim(P) == c(p,p) ) )
    stopifnot( length(m) == p )
    stopifnot( length(w.init) == p )
    stopifnot( length(b.init) == 1 )
    stopifnot( length(l1) == 1 )
    stopifnot( length(l2) == 1 )

    ## Verify name matching (if applicable)
    if( is.null(colnames(X)) == FALSE && is.null(colnames(P)) == FALSE )
      {
        stopifnot( is.null( rownames(P) ) == FALSE )
        stopifnot( all( colnames(X) == rownames(P) ) )
        stopifnot( all( colnames(X) == colnames(P) ) )
      }

    ## Set the initial parameter estimates
    S <- X %*% w.init + b.init
    Pw <- P %*% (w.init - m)

    NULL
  }

#' GELnet for logistic regression
#'
#' Constructs a GELnet model for logistic regression using the Newton method.
#'
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the linear regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of binary response labels (must be in {0,1})
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param b.init initial parameter estimate for the bias term
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being trained
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @return A list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model}
#' }
#' @noRd
gelnet.logreg <- function( X, y, l1, l2, d = rep(1,p), P = diag(p), m = rep(0,p),
                          max.iter = 100, eps = 1e-5, w.init = rep(0,p),
                          b.init = 0, silent = FALSE, balanced = FALSE, nonneg = FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Verify argument dimensionality
    stopifnot( sort(unique(y)) == c(0,1) )
    stopifnot( length(y) == n )
    stopifnot( length(d) == p )
    stopifnot( all( dim(P) == c(p,p) ) )
    stopifnot( length( w.init ) == p )
    stopifnot( length( b.init ) == 1 )
    stopifnot( length(l1) == 1 )
    stopifnot( length(l2) == 1 )

    ## Verify name matching (if applicable)
    if( is.null(colnames(X)) == FALSE && is.null(colnames(P)) == FALSE )
      {
        stopifnot( is.null( rownames(P) ) == FALSE )
        stopifnot( all( colnames(X) == rownames(P) ) )
        stopifnot( all( colnames(X) == colnames(P) ) )
      }

    ## Set the initial parameter estimates
    S <- X %*% w.init + b.init
    Pw <- P %*% (w.init - m)

    NULL
  }

#' GELnet for one-class regression
#'
#' Constructs a GELnet model for one-class regression using the Newton method.
#'
#' The function optimizes the following objective:
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i }
#' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the linear regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @return A list with one element:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#' }
#' @noRd
gelnet.oneclass <- function( X, l1, l2, d = rep(1,p), P = diag(p), m = rep(0,p),
                            max.iter = 100, eps = 1e-5,
                            w.init = rep(0,p), silent= FALSE, nonneg=FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Verify argument dimensionality
    stopifnot( length(d) == p )
    stopifnot( all( dim(P) == c(p,p) ) )
    stopifnot( length(m) == p )
    stopifnot( length(w.init) == p )
    stopifnot( length(l1) == 1 )
    stopifnot( length(l2) == 1 )

    ## Verify name matching (if applicable)
    if( is.null(colnames(X)) == FALSE && is.null(colnames(P)) == FALSE )
      {
        stopifnot( is.null( rownames(P) ) == FALSE )
        stopifnot( all( colnames(X) == rownames(P) ) )
        stopifnot( all( colnames(X) == colnames(P) ) )
      }

    ## Set the initial parameter estimates
    w <- w.init

    ## Run Newton's method
    fprev <- gelnet_oclr_obj( w, X, l1, l2, d, P, m )
    for( iter in 1:max.iter )
      {
        if( silent == FALSE )
          {
            cat( "Iteration", iter, ": " )
            cat( "f =", fprev, "\n" )
          }

        ## Compute the current fit
        s <- X %*% w
        pr <- 1 / ( 1 + exp(-s) )

        ## Compute the sample weights and active response
        a <- pr * (1-pr)
        z <- s + 1/pr

        ## Run coordinate descent for the resulting regression problem
        mm <- gelnet_lin_opt( X, z, l1, l2, a=a, d=d, P=P, m=m, max_iter = iter*2,
                             eps=eps, w_init = w, b_init = 0, fix_bias = TRUE,
                             silent = TRUE, nonneg=nonneg )
        w <- mm$w

        f <- gelnet_oclr_obj( w, X, l1, l2, d, P, m )
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else fprev <- f
      }

    list( w = w )
  }

#' A GELnet model with a requested number of non-zero weights
#'
#' Binary search to find an L1 penalty parameter value that yields the desired
#'   number of non-zero weights in a GELnet model.
#'
#' The method performs simple binary search starting in [0, l1s] and iteratively
#' training a model using the provided \code{f.gelnet}. At each iteration, the
#' method checks if the number of non-zero weights in the model is higher or lower
#' than the requested \code{nF} and adjusts the value of the L1 penalty term accordingly.
#' For linear regression problems, it is recommended to initialize \code{l1s} to the output
#' of \code{L1.ceiling}.
#'
#' @param f.gelnet a function that accepts one parameter: L1 penalty value,
#'    and returns a typical GELnets model (list with w and b as its entries)
#' @param nF the desired number of non-zero features
#' @param l1s the right side of the search interval: search will start in [0, l1s]
#' @param max.iter the maximum number of iterations of the binary search
#'
#' @return The model with the desired number of non-zero weights and the corresponding value of the
#' L1-norm parameter. Returned as a list with three elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model}
#'   \item{l1}{scalar, the corresponding value of the L1-norm parameter}
#' }
#' @seealso \code{\link{L1.ceiling}}
#' @noRd
gelnet.L1bin <- function( f.gelnet, nF, l1s, max.iter=10 )
  {
    ## Set up the search region
    L1top <- l1s
    L1bot <- 0

    ## Perform binary search
    for( i in 1:max.iter )
      {
        cat( "Binary search iteration", i, "\n" )
        l1 <- (L1top + L1bot) / 2
        m <- f.gelnet( l1 )
        k <- sum( m$w != 0 )
        cat( "Learned a model with", k, "non-zero features\n" )
        if( k == nF ) break
        if( k < nF ) {L1top <- l1} else {L1bot <- l1}
      }

    ## Store the selected L1 parameter value into the model
    m$l1 <- l1
    m
  }

