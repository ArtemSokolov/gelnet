## Generalized Elastic Nets
##
## by Artem Sokolov

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
#' @param max_iter maximum number of iterations
#' @param eps convergence precision
#' @param w_init initial parameter estimate for the weights
#' @param b_init initial parameter estimate for the bias term
#' @param fix_bias set to TRUE to prevent the bias term from being updated (regression only) (default: FALSE)
#' @param silent set to TRUE to suppress run-time output to stdout; overrides verbose (default: FALSE)
#' @param verbose set to TRUE to see extra output; is overridden by silent (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @return A list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model (omitted for one-class models)}
#' }
#' @export
gelnet_train <- function( X, y, l1, l2, nFeats=NULL, a=NULL, d=NULL, P=NULL, m=NULL,
                         max_iter=100, eps=1e-5, w_init=rep(0,p), b_init=NULL,
                         fix_bias=FALSE, silent=FALSE, verbose=FALSE, balanced=FALSE, nonneg=FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## One-class
    if( is.null(y) )
    {
        if( !silent ) cat( "Training a one-class model\n" )
        f.gel <- function(L1)
        {gelnet_oclr_opt( X, L1, l2, max_iter, eps, silent, verbose, nonneg, w_init, d, P, m )}
    }

    ## Binary classification
    else if( is.factor(y) )
      {
          if( nlevels(y) == 1 )
              stop( "All labels are identical\nConsider training a one-class model instead" )
          if( nlevels(y) > 2 )
              stop( "Labels belong to a multiclass task\nConsider training a set of one-vs-one or one-vs-rest models" )
          if( !silent ) cat( "Training a logistic regression model\n" )
          if( is.null(b_init) ) b_init <- 0

          ## Convert the labels to {0,1}
          y <- factor(y)
          if( !silent )
              cat( "Treating", levels(y)[1], "as the positive class\n" )
          y <- as.integer( y == levels(y)[1] )

          f.gel <- function(L1)
          {gelnet_blr_opt( X, y, L1, l2, max_iter, eps, silent, verbose,
                          balanced, nonneg, w_init, b_init, d, P, m )}
      }

    ## Regression
    else if( is.numeric(y) )
      {
          if( !silent ) cat( "Training a linear regression model\n" )
          if( is.null(b_init) ) b_init <- sum(a*y) / sum(a)
          f.gel <- function(L1)
          { gelnet_lin_opt( X, y, L1, l2, max_iter, eps, fix_bias, silent, verbose,
                           nonneg, w_init, b_init, a, d, P, m) }
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

