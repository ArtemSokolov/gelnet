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

    ## Handle alternative L1 parameterization via desired number of features
    if( "nFeats" %in% names(modeldef) )
    {
        ## Set up a standard model definition halfway to the L1 ceiling
        l1top <- L1_ceiling(modeldef)
        l1bot <- 0
        mdf <- modeldef
        mdf$nFeats <- NULL

        for( iter in 1:max_iter )
        {
            if( !silent ) cat( "Binary search iteration", iter, ":" )

            ## Train the model and adjust the L1 coefficient based on
            ##    the number of non-zero weights
            mdf$l1 <- (l1top + l1bot) / 2
            mdl <- gelnet_train( mdf, max_iter, eps, silent=TRUE )
            nf <- sum(mdl$w != 0)
            if( !silent ) cat( "model with", nf, "non-zero weights\n" )
            if( nf == modeldef$nFeats )
            {
                if( verbose ) cat( "Final L1 coefficient:", mdf$l1, "\n" )
                return(mdl)
            }
            if( nf < modeldef$nFeats ) {l1top <- mdf$l1} else {l1bot <- mdf$l1}
        }

        stop( "Unable to reach ", modeldef$nFeats, " non-zero features within ",
             max_iter, " iterations" )
    }

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

