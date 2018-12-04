## Composition of model definitions
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

## Composition with supervised task definition
## Not exported
gelnetComposite.taskdef_sv <- function( term, mdl )
{
    if( nrow(mdl$X) != length(term$y) )
        stop( "Number of labels does not match the number of samples" )
    UseMethod( "gelnetComposite", NULL )
}

## Composition with L1 regularizer
## Not exported
gelnetComposite.rglzdef_L1 <- function( term, mdl )
{
    if( !is.null(term$d) && ncol(mdl$X) != length(term$d) )
        stop( "Number of feature weights does not match the number of features" )
    UseMethod( "gelnetComposite", NULL )
}

## Composition with L2 regularizer
## Not exported
gelnetComposite.rglzdef_L2 <- function( term, mdl )
{
    if( !is.null(term$P) && ncol(term$P) != ncol(mdl$X) )
        stop( "Penalty matrix dimensionality does not match the number of features" )
    if( !is.null(term$m) && length(term$m) != ncol(mdl$X) )
        stop( "Number of translation coefficients does not match the number of features" )
    UseMethod( "gelnetComposite", NULL )
}

## Composition with initializer
## Not exported
gelnetComposite.gelinit <- function( term, mdl )
{
    if( !is.null(term$w_init) && length(term$w_init) != ncol(mdl$X) )
        stop( "The number of weights provided to the initializer does not match the number of weights in the model" )
    UseMethod( "gelnetComposite", NULL )
}
