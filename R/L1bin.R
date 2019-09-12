## Binary search along the L1 parameter
##
## by Artem Sokolov

#' The largest meaningful value of the L1 parameter
#'
#' Computes the smallest value of the LASSO coefficient L1 that leads to an
#'  all-zero weight vector for a given gelnet model
#'
#' The cyclic coordinate descent updates the model weight \eqn{w_k} using a soft threshold operator
#' \eqn{ S( \cdot, \lambda_1 d_k ) } that clips the value of the weight to zero, whenever the absolute
#' value of the first argument falls below \eqn{\lambda_1 d_k}. From here, it is straightforward to compute
#' the smallest value of \eqn{\lambda_1}, such that all weights are clipped to zero.
#'
#' @param modeldef model definition constructed through gelnet() arithmetic
#' @return The largest meaningful value of the L1 parameter (i.e., the smallest value that yields a model with all zero weights)
#' @export
L1_ceiling <- function( modeldef )
{

    ## Direct the call to the appropriate C++ function
    if( is.null(modeldef$y) )		
    {
        ## One-class model
        l1c_oclr( modeldef$X, modeldef$l2,
                 modeldef$d, modeldef$P, modeldef$m )
    }
    else if( is.factor(modeldef$y) )
    {
        ## Binary model
        ## Convert the labels to {0,1}
        y <- with( modeldef, as.integer(y == levels(y)[1]) )
        l1c_blr( modeldef$X, y, modeldef$l2, modeldef$balanced,
                modeldef$d, modeldef$P, modeldef$m )
    }
    else if( is.numeric(modeldef$y) )
    {
        ## Liner regression model
        l1c_lin( modeldef$X, modeldef$y, modeldef$l2, modeldef$a,
                modeldef$d, modeldef$P, modeldef$m )
    }
    else
    { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }
  }

