## Generates small datasets for testthat tests
##
## by Artem Sokolov

## Generates data for linear regression testing
gen_lin <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset a seed to ensure reproducibility
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( l1 = 0.1, l2 = 10, z = rnorm(n), X = matrix(rnorm(n*p), n, p),
                   a = runif(n), d = runif(p), P = t(A)%*%A/p, m = rnorm(p, sd=0.1) )
    save( params, file="lin.RData" )
}

## Generates data for binary logistic regression testing
gen_blr <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset a seed to ensure reproducibility
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( l1 = 0.1, l2 = 10, y = sample(c(0,1), n, replace=TRUE),
                   X = matrix(rnorm(n*p), n, p), d = runif(p), P = t(A)%*%A/p,
                   m = rnorm(p, sd=0.1) )
    save( params, file="blr.RData" )
}

## Generates data for one-class logistic regression testing
gen_oclr <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset a seed to ensure reproducibility
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( l1 = 0.1, l2 = 10, X = matrix(rnorm(n*p), n, p),
                   d = runif(p), P = t(A)%*%A/p, m = rnorm(p, sd=0.1) )
    save( params, file="oclr.RData" )
}
