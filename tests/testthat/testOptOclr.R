context( "One-class logistic regression model training" )

source( "custom.R" )

## Generates parameters for one-class logistic regression testing
gen_params_oclr <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- purrr::list_modify( params[[1]], d = runif(p) )
    params[[3]] <- purrr::list_modify( params[[2]], P = t(A)%*%A/p )
    params[[4]] <- purrr::list_modify( params[[3]], m = rnorm(p, sd=0.1) )
    params
}

## Generates model definitions based on the provided set of parameters
gen_modeldef_oclr <- function( params )
{
    dd <- list()
    dd[[1]] <- gelnet( params[[1]]$X ) + rglz_L1( params[[1]]$l1 ) + rglz_L2( params[[2]]$l2 )
    dd[[2]] <- dd[[1]] + rglz_L1( params[[2]]$l1, params[[2]]$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params[[3]]$l2, params[[3]]$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params[[4]]$l2, params[[4]]$P, params[[4]]$m )
    dd[[5]] <- dd[[4]] + model_oclr(TRUE)
    dd
}

test_that( "One-class logistic regression training", {
    ## Silently trains a logistic GELnet model using the provided parameters
    ftrain <- gen_ftrain( gelnet_oclr_opt )

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
    { function( mdl ) { do.call( gelnet_oclr_obj, c(mdl,prms) ) } }

    ## Generate the models and matching objective functions
    params <- gen_params_oclr()
    mm <- purrr::map( params, ftrain )
    ff <- purrr::map( params, fgen )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 17 )
    expect_equal( mm[[1]]$w[26], -0.01308364, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal, FALSE )
    expect_relopt( mm, ff )

    ## Test non-negativity
    mm[[5]] <- ftrain( params[[4]], silent=TRUE, nonneg=TRUE )
    purrr::map( mm[[5]]$w, expect_gte, 0 )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[5]]) )

    ## Compose model definitions using the "grammar of modeling"
    dd <- gen_modeldef_oclr( params )

    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})
