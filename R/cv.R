## Cross-validation
##
## by Artem Sokolov

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
gelnet_cv <- function( X, y, nL1, nL2, nFolds=5, a=rep(1,n), d=rep(1,p), P=diag(p), m=rep(0,p),
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
                mm <- gelnet_train( X[j.tr,], y[j.tr], l1, l2, NULL, a[j.tr], d, P, m,
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
