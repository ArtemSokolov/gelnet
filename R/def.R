## Model definitions
##
## by Artem Sokolov

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
#' @param nonneg set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )
#' @param balanced boolean specifying whether the balanced model is being trained (binary classification only) (default: FALSE)
#' @return A GELnet model definition
#' @export
gelnet <- function( X, y=NULL, nonneg=FALSE, balanced=FALSE )
{
    ## Check argument dimensionality
    if( nrow(X) != length(y) )
        stop( "Number of labels does not match the number of samples" )

    ## Create a bare-bones model
    mdl <- list( X=X, nonneg=nonneg )
    class( mdl ) <- "geldef"

    ## One-class
    if( is.null(y) )
    {
        if( balanced ) warning( "Ignoring balanced setting in one-class models" )
    }

    ## Binary classification
    else if( is.factor(y) || is.character(y) )
    {
        y <- factor(y)		## Handles the non-factor character vectors

        if( nlevels(y) == 1 )
            stop( "All labels are identical\n  Consider training a one-class model instead" )
        if( nlevels(y) > 2 )
            stop( paste0("Labels belong to a multiclass task\n  ",
                         "Consider training a set of one-vs-one or one-vs-rest models") )

        ## Convert the labels to {0,1}
        mdl$y <- as.integer( y == levels(y)[1] )
        mdl$balanced <- balanced
    }

    ## Linear regression
    else if( is.numeric(y) )
    {
        if( balanced ) warning( "Ignoring balanced setting in linear regression models" )
        mdl$y <- y
    }

    ## Other
    else
    { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }

    mdl
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
gel_L1 <- function( l1, d=NULL )
{
    rglz <- list( l1=l1, d=d )
    class(rglz) <- "geldefL1"
    rglz
}

## Composition + operator for GELnet model definition
`+.geldef` <- function( lhs, rhs )
    { gelnetComposite( rhs, lhs ) }

## S3 generic for model composition
## Not exported
gelnetComposite <- function( term, ... )
    { UseMethod( "gelnetComposite" ) }

## Composite function for gel_L1()
## Not exported
gelnetComposite.geldefL1 <- function( rglz, mdl )
{
    mdl$l1 <- rglz$l1
    mdl$d <- rglz$d
    mdl
}
