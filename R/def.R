## Model definitions
##
## by Artem Sokolov

#' Composition operator for GELnet model definition
#'
#' @param lhs left-hand side of composition (current chain)
#' @param rhs right-hand side of composition (new module)
#' @export
`+.geldef` <- function( lhs, rhs )
{
    if( class(lhs) == "geldef" ) gelnetComposite( rhs, lhs )
    else gelnetComposite( lhs, rhs )
}

## S3 generic for model composition
## Not exported
gelnetComposite <- function( term, ... )
{ UseMethod( "gelnetComposite" ) }

## Default behavior for term composition
## Not exported
gelnetComposite.default <- function( term, mdl )
{
    for( i in names(term) )
        mdl[[i]] <- term[[i]]
    mdl
}

#' GELnet model definition for linear regression, binary classification or one-class problems.
#'
#' Infers the problem type and constructs the appropriate GELnet model definition.
#'
#' The method determines the problem type from the labels argument y.
#' If y is a numeric vector, then a regression model is defined which optimizes the following objective function:
#' \deqn{ \frac{1}{2n} \sum_i a_i (y_i - (w^T x_i + b))^2 + R(w) }
#'
#' If y is a factor with two levels, then the function returns a binary classification model definition, which optimizes the following objective function:
#' \deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i + b }
#'
#' Finally, if no labels are provided (y == NULL), then a one-class model is constructed using the following objective function:
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i }
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of response values. Must be numeric vector for regression, factor with 2 levels for binary classification, or NULL for a one-class task.
#' @param a n-by-1 vector of sample weights (regression only)
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @return A GELnet model definition
#' @export

gelnet <- function( X )
{
    if( !is.matrix(X) )
        stop( "Argument must be a matrix" )
    structure( list(X=X, l1=0, l2=0), class = "geldef" )
}

model_oclr <- function( nonneg=FALSE )
{ structure( list(nonneg=nonneg), class = "taskdef" ) }

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

gelnetComposite.taskdef_sv <- function( term, mdl )
{
    if( nrow(mdl$X) != length(term$y) )
        stop( "Number of labels does not match the number of samples" )
    UseMethod( "gelnetComposite", NULL )
}

model_lin <- function( y, a=NULL, nonneg=FALSE )
{
    if( !is.numeric(y) ) stop( "Labels must be numeric" )
    if( !is.null(a) && length(a) != length(y) )
        stop( "The number of sample weights must match the number of labels" )
    structure( list(y=y, a=a, nonneg=nonneg), class = "taskdef_sv" )
}

#' L1 regularizer for GELnet models
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
    class(rglz) <- "geldefL1"
    rglz
}

#' L2 regularizer for GELnet models
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
    rglz <- list( l2=l2, P=P, m=m )
    class(rglz) <- "geldefL2"
    rglz
}

#' Initializer for GELnet models
#'
#' Defines initial values for model weights and the bias term
#'
#' @param w_init p-by-1 vector of initial weight values
#' @param b_init scalar, initial value for the bias term
#' @export
gel_init <- function( w_init, b_init )
{
    initr <- list( w_init=w_init, b_init=b_init )
    class(initr) <- "gelinit"
    initr
}

