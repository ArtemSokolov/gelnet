## Model definitions
##
## by Artem Sokolov

gelnet <- function( X, y=NULL, nonneg=FALSE, balanced=FALSE )
{
    mdl <- list( X=X, nonneg=nonneg )
    
    ## One-class
    if( is.null(y) )
    {
        if( balanced ) warning( "Ignoring balanced setting in one-class models" )
    }

    ## Binary classification
    else if( is.factor(y) )
    {
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

    class( mdl ) <- "geldef"
    mdl
}
