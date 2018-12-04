## Binary search along the L1 parameter
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
#' @param max_iter the maximum number of iterations of the binary search
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
gelnet.L1bin <- function( f.gelnet, nF, l1s, max_iter=10 )
  {
    ## Set up the search region
    L1top <- l1s
    L1bot <- 0

    ## Perform binary search
    for( i in 1:max_iter )
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
