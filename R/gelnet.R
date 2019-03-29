## Generalized Elastic Nets
##
## by Artem Sokolov

#' Trains a GELnet model
#'
#' Trains a model on the definition constructed by gelnet()
#' 
#' The training is performed through cyclical coordinate descent, and the optimization is terminated after the desired tolerance is achieved or after a maximum number of iterations.
#'
#' @param modeldef model definition constructed through gelnet() arithmetic
#' @param max_iter maximum number of iterations
#' @param eps convergence precision
#' @param silent set to TRUE to suppress run-time output to stdout; overrides verbose (default: FALSE)
#' @param verbose set to TRUE to see extra output; is overridden by silent (default: FALSE)
#' @return A GELNET model, expressed as a list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model (omitted for one-class models)}
#' }
#' @export
gelnet_train <- function( modeldef, max_iter = 100L, eps = 1e-5, silent=FALSE, verbose=FALSE )
{
    if( class(modeldef) != "geldef" )
        stop( "Please provide a model definition created via gelnet()" )

    ## Compose the list of parameters
    params <- modeldef + list( max_iter=max_iter, eps=eps, silent=silent, verbose=verbose )
    
    ## One-class logistic regression
    if( is.null(params$y) )
    {
        if( !silent ) cat( "Training a one-class model\n" )
        do.call( gelnet_oclr_opt, params )
    }

    ## Binary logistic regression
    else if( is.factor(params$y) )
    {
        if( !silent ) cat( "Training a binary logistic regression model\n" )
        
        ## Convert the labels to {0,1}
        if( verbose ) cat( "Treating", levels(params$y)[1], "as the positive class\n" )
        params$y <- as.integer( params$y == levels(params$y)[1] )
        
        do.call( gelnet_blr_opt, params )
    }

    ## Linear regression
    else if( is.numeric(params$y) )
    {
        if( !silent ) cat( "Training a linear regression model\n" )
        params$z <- params$y
        params$y <- NULL
        do.call( gelnet_lin_opt, params )
    }

    else
    { stop( "Unknown label type\ny must be a numeric vector, a 2-level factor or NULL" ) }
}

