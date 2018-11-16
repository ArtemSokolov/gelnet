## Kernel methods for Generalized Elastic Nets
##
## by Artem Sokolov

#' Kernel models for linear regression, binary classification and one-class problems.
#'
#' Infers the problem type and learns the appropriate kernel model.
#'
#' The entries in the kernel matrix K can be interpreted as dot products
#' in some feature space \eqn{\phi}. The corresponding weight vector can be
#' retrieved via \eqn{w = \sum_i v_i \phi(x_i)}. However, new samples can be
#' classified without explicit access to the underlying feature space:
#' \deqn{w^T \phi(x) + b = \sum_i v_i \phi^T (x_i) \phi(x) + b = \sum_i v_i K( x_i, x ) + b}
#'
#' The method determines the problem type from the labels argument y.
#' If y is a numeric vector, then a ridge regression model is trained by optimizing the following objective function:
#' \deqn{ \frac{1}{2n} \sum_i a_i (z_i - (w^T x_i + b))^2 + w^Tw }
#'
#' If y is a factor with two levels, then the function returns a binary classification model, obtained by optimizing the following objective function:
#' \deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + w^Tw }
#'  where
#' \deqn{ s_i = w^T x_i + b }
#'
#' Finally, if no labels are provided (y == NULL), then a one-class model is constructed using the following objective function:
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + w^Tw }
#'  where
#' \deqn{ s_i = w^T x_i }
#'
#' In all cases, \eqn{w = \sum_i v_i \phi(x_i)} and the method solves for \eqn{v_i}.
#'
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param y n-by-1 vector of response values. Must be numeric vector for regression, factor with 2 levels for binary classification, or NULL for a one-class task.
#' @param lambda scalar, regularization parameter
#' @param a n-by-1 vector of sample weights (regression only)
#' @param max.iter maximum number of iterations (binary classification and one-class problems only)
#' @param eps convergence precision (binary classification and one-class problems only)
#' @param v.init initial parameter estimate for the kernel weights (binary classification and one-class problems only)
#' @param b.init initial parameter estimate for the bias term (binary classification only)
#' @param fix.bias set to TRUE to prevent the bias term from being updated (regression only) (default: FALSE)
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @return A list with two elements:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#'   \item{b}{scalar, bias term for the linear model (omitted for one-class models)}
#' }
#' @seealso \code{\link{gelnet}}
#' @export
gelnet.ker <- function( K, y, lambda, a, max.iter = 100, eps = 1e-5, v.init=rep(0,nrow(K)),
                       b.init=0, fix.bias=FALSE, silent=FALSE, balanced=FALSE )
  {
    ## Determine the problem type
    if( is.null(y) )
      {
        if( !silent ) cat( "Training a one-class model\n" )
        gelnet.kor( K, lambda, max.iter, eps, v.init, silent )
      }
    else if( is.factor(y) )
      {
        if( nlevels(y) == 1 )
          stop( "All labels are identical\nConsider training a one-class model instead" )
        if( nlevels(y) > 2 )
          stop( "Labels belong to a multiclass task\nConsider training a set of one-vs-one or one-vs-rest models" )
        if( !silent ) cat( "Training a logistic regression model\n" )
        gelnet.klr( K, y, lambda, max.iter, eps, v.init, b.init, silent, balanced )
      }
    else if( is.numeric(y) )
      {
        if( !silent ) cat( "Training a linear regression model\n" )
        if( b.init == 0 ) b.init <- sum(a*y) / sum(a)
        gelnet.krr( K, y, lambda, a, fix.bias )
      }
    else
      { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }
  }

#' Kernel ridge regression
#'
#' Learns a kernel ridge regression model.
#'
#' The entries in the kernel matrix K can be interpreted as dot products
#' in some feature space \eqn{\phi}. The corresponding weight vector can be
#' retrieved via \eqn{w = \sum_i v_i \phi(x_i)}. However, new samples can be
#' classified without explicit access to the underlying feature space:
#' \deqn{w^T \phi(x) + b = \sum_i v_i \phi^T (x_i) \phi(x) + b = \sum_i v_i K( x_i, x ) + b}
#'
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param y n-by-1 vector of response values
#' @param lambda scalar, regularization parameter
#' @param a n-by-1 vector of samples weights
#' @param fix.bias set to TRUE to force the bias term to 0 (default: FALSE)
#' @return A list with two elements:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#'   \item{b}{scalar, bias term for the model}
#' }
#' @noRd
gelnet.krr <- function( K, y, lambda, a, fix.bias=FALSE )
  {
    ## Argument verification
    n <- nrow(K)
    stopifnot( length(y) == n )
    stopifnot( length(a) == n )
    stopifnot( is.null(dim(a)) )	## a must not be in matrix form

    ## Set up the sample weights and kernalization of the bias term
    A <- diag(a)

    if( fix.bias )
      {
        ## Solve the optimization problem in closed form
        m <- solve( K %*% A %*% t(K) + lambda * n * K, K %*% A %*% y )
        list( v = drop(m), b = 0 )
      }
    else
      {
        ## Set up kernalization of the bias term
        K1 <- rbind( K, 1 )
        K0 <- cbind( rbind( K, 0 ), 0 )

        ## Solve the optimization problem in closed form
        m <- solve( K1 %*% A %*% t(K1) + lambda * n * K0, K1 %*% A %*% y )

        ## Retrieve the weights and the bias term
        list( v = m[1:n], b = m[n+1] )
      }
  }

#' Kernel logistic regression
#'
#' Learns a kernel logistic regression model for a binary classification task
#'
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the kernel ridge regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param y n-by-1 vector of binary response labels
#' @param lambda scalar, regularization parameter
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param v.init initial parameter estimate for the kernel weights
#' @param b.init initial parameter estimate for the bias term
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being evaluated
#' @return A list with two elements:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#'   \item{b}{scalar, bias term for the model}
#' }
#' @seealso \code{\link{gelnet.krr}}, \code{\link{gelnet.logreg.obj}}
#' @noRd
gelnet.klr <- function( K, y, lambda, max.iter = 100, eps = 1e-5,
                       v.init=rep(0,nrow(K)), b.init=0, silent=FALSE, balanced=FALSE )
  {
    ## Argument verification
    stopifnot( nrow(K) == ncol(K) )
    stopifnot( nrow(K) == length(y) )

    ## Convert the labels to {0,1}
    y <- factor(y)
    if( !silent )
      cat( "Treating", levels(y)[1], "as the positive class\n" )
    y <- as.integer( y == levels(y)[1] )

    ## Objective function to minimize
    fobj <- function( v, b )
      {
        s <- K %*% v + b
        R2 <- lambda * t(v) %*% K %*% v / 2
        LL <- mean( y * s - log(1+exp(s)) )
        R2 - LL
      }

    ## Compute the initial objective value
    v <- v.init
    b <- b.init
    fprev <- fobj( v, b )

    ## Reduce kernel logistic regression to kernel regression
    ##  about the current Taylor method estimates
    for( iter in 1:max.iter )
      {
        if( silent == FALSE )
          {
            cat( "Iteration", iter, ": " )
            cat( "f =", fprev, "\n" )
          }

        ## Compute the current fit
        s <- drop(K %*% v + b)
        pr <- 1 / (1 + exp(-s))

        ## Snap probabilities to 0 and 1 to avoid division by small numbers
        j0 <- which( pr < eps )
        j1 <- which( pr > (1-eps) )
        pr[j0] <- 0; pr[j1] <- 1

        ## Compute the sample weights and the response
        a <- pr * (1-pr)
        a[c(j0,j1)] <- eps
        z <- s + (y - pr) / a

        ## Rebalance the sample weights according to the class counts
        if( balanced )
          {
            jpos <- which( y==1 )
            jneg <- which( y==0 )
            a[jpos] <- a[jpos] * length(y) / ( length(jpos)*2 )
            a[jneg] <- a[jneg] * length(y) / ( length(jneg)*2 )
          }

        ## Run the coordinate descent for the resulting regression problem
        m <- gelnet.krr( K, z, lambda, a )
        v <- m$v
        b <- m$b

        ## Evaluate the objective function and check convergence criteria
        f <- fobj( v, b )
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else fprev <- f
      }

    list( v = v, b = b )
  }

#' Kernel one-class regression
#'
#' Learns a kernel one-class model for a given kernel matrix
#'
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the kernel ridge regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#'
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param lambda scalar, regularization parameter
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param v.init initial parameter estimate for the kernel weights
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @return A list with one element:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#' }
#' @seealso \code{\link{gelnet.krr}}, \code{\link{gelnet.oneclass.obj}}
#' @noRd
gelnet.kor <- function( K, lambda, max.iter = 100, eps = 1e-5,
                       v.init = rep(0,nrow(K)), silent= FALSE )
  {
    ## Argument verification
    stopifnot( nrow(K) == ncol(K) )

    ## Define the objective function to optimize
    fobj <- function( v )
      {
        s <- K %*% v
        LL <- mean( s - log( 1 + exp(s) ) )
        R2 <- lambda * t(v) %*% K %*% v / 2
        R2 - LL
      }

    ## Set the initial parameter estimates
    v <- v.init
    fprev <- fobj( v )

    ## Run Newton's method
    for( iter in 1:max.iter )
      {
        if( silent == FALSE )
          {
            cat( "Iteration", iter, ": " )
            cat( "f =", fprev, "\n" )
          }

        ## Compute the current fit
        s <- drop(K %*% v)
        pr <- 1 / (1 + exp(-s))

        ## Compute the sample weights and active response
        a <- pr * (1-pr)
        z <- s + 1/pr

        ## Solve the resulting regression problem
        mm <- gelnet.krr( K, z, lambda, a, fix.bias=TRUE )
        v <- mm$v

        ## Compute the new objective function value
        f <- fobj( v )
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else fprev <- f
      }

    list( v = v )
  }
