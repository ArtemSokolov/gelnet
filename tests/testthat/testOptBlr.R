context( "Binary logistic regression model training" )

source( "custom.R" )

## Generates parameters for binary logistic regression testing
gen_params_blr <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, y = sample( c(0,1), n, replace=TRUE ),
                         X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- purrr::list_modify( params[[1]], d = runif(p) )
    params[[3]] <- purrr::list_modify( params[[2]], P = t(A)%*%A/p )
    params[[4]] <- purrr::list_modify( params[[3]], m = rnorm(p, sd=0.1) )
    params[[5]] <- purrr::list_modify( params[[4]], balanced=TRUE )
    params
}

## Generates model definitions based on the provided set of parameters
gen_modeldef_blr <- function( params )
{
    dd <- list()
    dd[[1]] <- gelnet(params[[1]]$X) + model_blr(factor(params[[1]]$y, c(1,0))) +
        rglz_L1( params[[1]]$l1 ) + rglz_L2( params[[1]]$l2 )
    dd[[2]] <- dd[[1]] + rglz_L1( params[[2]]$l1, params[[2]]$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params[[3]]$l2, params[[3]]$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params[[4]]$l2, params[[4]]$P, params[[4]]$m )
    dd[[5]] <- dd[[4]] + model_blr( factor(params[[5]]$y, c(1,0)), balanced=TRUE )
    dd[[6]] <- dd[[5]] + model_blr( factor(params[[5]]$y, c(1,0)), TRUE, TRUE )
    dd
}

test_that( "Binary logistic regression training", {
    ## Silently trains a logistic GELnet model using the provided parameters
    ftrain <- gen_ftrain( gelnet_blr_opt )

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
    { function( mdl ) { do.call( gelnet_blr_obj, c(mdl,prms) ) } }

    ## Generate the models and matching objective functions
    params <- gen_params_blr()
    mm <- purrr::map( params, ftrain )
    ff <- purrr::map( params, fgen )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 21 )
    expect_equal( mm[[1]]$b, 0.400053, tol=1e-5 )
    expect_equal( mm[[1]]$w[21], -0.005406831, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal )
    expect_relopt( mm, ff )

    ## Balanced model should have bias term closer to 0
    expect_lt( abs(mm[[5]]$b), abs(mm[[4]]$b) )

    ## Test non-negativity
    mm[[6]] <- ftrain( params[[5]], nonneg=TRUE, silent=TRUE )
    purrr::map( mm[[6]]$w, expect_gte, 0 )
    expect_lt( ff[[5]](mm[[5]]), ff[[5]](mm[[6]]) )
    
    ## Compose model definitions using the "grammar of modeling"
    dd <- gen_modeldef_blr( params )
    
    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )

    ## Test the L1 ceiling computation
    
})

