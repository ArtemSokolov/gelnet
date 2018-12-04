## Model definitions
##
## by Artem Sokolov

#' GELnet model definition
#'
#' Starting building block for defining a GELnet model
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @return A GELnet model definition
#' @export
gelnet <- function( X )
{
    if( !is.matrix(X) )
        stop( "Argument must be a matrix" )
    structure( list(X=X, l1=0, l2=0), class = "geldef" )
}

#' One-class logistic regression
#'
#' Defines a one-class logistic regression (OCLR) task
#'
#' The OCLR objective function is defined as
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i }
#'
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE)
#' @return A GELnet task definition that can be combined with gelnet() output
#' @export
model_oclr <- function( nonneg=FALSE )
{ structure( list(nonneg=nonneg), class = "taskdef" ) }

#' Binary logistic regression
#'
#' Defines a binary logistic regression task
#'
#' The binary logistic regression objective function is defined as
#' \deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i + b }
#' @param y n-by-1 factor with two levels
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE)
#' @param balanced boolean specifying whether the balanced model is being trained (default: FALSE)
#' @return A GELnet task definition that can be combined with gelnet() output
#' @export
model_blr <- function( y, nonneg=FALSE, balanced=FALSE )
{
    ## Hamdle non-factor character vectors
    y <- factor(y)

    if( nlevels(y) == 1 )
        stop( "All labels are identical\n  Consider training a one-class model instead" )
    if( nlevels(y) > 2 )
        stop( paste0("Labels belong to a multiclass task\n  ",
                     "Consider training a set of one-vs-one or one-vs-rest models") )

    structure( list(y=y, nonneg=nonneg, balanced=balanced), class = "taskdef_sv" )
}

#' Linear regression
#'
#' Defines a linear regression task
#'
#' The objective function is given by
#' \deqn{ \frac{1}{2n} \sum_i a_i (y_i - (w^T x_i + b))^2 + R(w) }
#' @param y n-by-1 numeric vector of response values
#' @param a n-by-1 vector of sample weights
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE)
#' @return A GELnet task definition that can be combined with gelnet() output
#' @export
model_lin <- function( y, a=NULL, nonneg=FALSE )
{
    if( !is.numeric(y) ) stop( "Labels must be numeric" )
    if( !is.null(a) && length(a) != length(y) )
        stop( "The number of sample weights must match the number of labels" )
    structure( list(y=y, a=a, nonneg=nonneg), class = "taskdef_sv" )
}

#' L1 regularizer
#'
#' Defines an L1 regularizer with optional per-feature weights
#'
#' The L1 regularization term is defined by
#' \deqn{ R1(w) = \lambda_1 \sum_j d_j |w_j| }
#'
#' @param l1 coefficient for the L1-norm penalty
#' @param d p-by-1 vector of feature weights
#' @return A regularizer definition that can be combined with a model definition using + operator
#' @export
gel_L1 <- function( l1, d=NULL )
{
    if( l1 < 0 ) stop("L1 coefficient must be non-negative")
    rglz <- list( l1=l1, d=d )
    class(rglz) <- "rglzdef_L1"
    rglz
}

#' L2 regularizer
#'
#' Defines an L2 regularizer with optional feature-feature penalties and translation coefficients
#'
#' The L2 regularizer term is define by
#' \deqn{ R2(w) = \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#'
#' @param l2 coefficient for the L2-norm penalty
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @return A regularizer definition that can be combined with a model definition using + operator
#' @export
gel_L2 <- function( l2, P=NULL, m=NULL )
{
    if( l2 < 0 ) stop("L2 coefficient must be non-negative")
    if( !is.null(P) && nrow(P) != ncol(P) ) stop("Penalty matrix must be square")
    rglz <- list( l2=l2, P=P, m=m )
    class(rglz) <- "rglzdef_L2"
    rglz
}

#' Initializer for GELnet models
#'
#' Defines initial values for model weights and the bias term
#'
#' If an initializer is NULL, the values are computed automatically during training
#'
#' @param w_init p-by-1 vector of initial weight values
#' @param b_init scalar, initial value for the bias term
#' @export
gel_init <- function( w_init = NULL, b_init = NULL )
{
    initr <- list( w_init=w_init, b_init=b_init )
    class(initr) <- "gelinit"
    initr
}

