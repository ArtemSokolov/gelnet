library( gelnet )

## Linear regression models
oldLin <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, y = rnorm(n),
                         X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- c( params[[1]], list(a = runif( n ), d = runif( p )) )
    params[[3]] <- c( params[[2]], list( P = t(A) %*% A / p ) )
    params[[4]] <- c( params[[3]], list(m = rnorm(p, sd=0.1)) )

    ## Run each model by hand
    f <- function( p ) do.call( gelnet:::gelnet.lin, p )
    mmOld <- purrr::map( params, f )
}

## Logistic regression models
oldLR <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10,
                         y = factor(sample( c(0,1), n, replace=TRUE ), c(1,0)),
                         X = matrix(rnorm(n*p), n, p) ))
    params[[2]] <- c( params[[1]], list(d = runif( p )) )
    params[[3]] <- c( params[[2]], list(P = t(A) %*% A / p) )
    params[[4]] <- c( params[[3]], list(m = rnorm(p, sd=0.1)) )
    params[[5]] <- c( params[[4]], list(balanced=TRUE) )
    params[[6]] <- c( params[[5]], list(nonneg=TRUE) )

    ## Run each model by hand
    f <- function( p ) do.call( gelnet:::gelnet.logreg, p )
    lrOld <- purrr::map( params, f )
    save( lrOld, file="lrOld.RData" )
}
