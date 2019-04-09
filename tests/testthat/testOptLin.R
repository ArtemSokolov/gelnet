context( "Linear regression model training" )

source( "custom.R" )

## Generates parameters for linear regression testing
gen_params_lin <- function()
{
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, z = rnorm(n),
                         X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- purrr::list_modify( params[[1]], a = runif(n), d = runif(p) )
    params[[3]] <- purrr::list_modify( params[[2]], P = t(A)%*%A/p )
    params[[4]] <- purrr::list_modify( params[[3]], m = rnorm(p, sd=0.1) )
    params
}

## Generates model definitions based on the provided set of parameters
gen_modeldef_lin <- function( params )
{
    dd <- list()
    dd[[1]] <- gelnet( params[[1]]$X ) + model_lin( params[[1]]$z ) +
        rglz_L1( params[[1]]$l1 ) + rglz_L2( params[[1]]$l2 )
    dd[[2]] <- dd[[1]] + model_lin( params[[2]]$z, params[[2]]$a ) +
        rglz_L1( params[[2]]$l1, params[[2]]$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params[[3]]$l2, params[[3]]$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params[[4]]$l2, params[[4]]$P, params[[4]]$m )
    dd[[5]] <- dd[[4]] + model_lin( params[[4]]$z, params[[4]]$a, fix_bias=TRUE )
    dd[[6]] <- dd[[4]] + model_lin( params[[4]]$z, params[[4]]$a, nonneg=TRUE )
    dd
}

test_that( "Linear regression training", {
    ## Silently trains a linear GELnet model using the provided parameters
    ftrain <- gen_ftrain( gelnet_lin_opt )

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
        { function( mdl ) { do.call( gelnet_lin_obj, c(mdl,prms) ) } }

    ## Generate the models and matching objective functions
    params <- gen_params_lin()
    mm <- purrr::map( params, ftrain )
    ff <- purrr::map( params, fgen )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 30 )
    expect_equal( mm[[1]]$b, 0.06710631, tol=1e-5 )
    expect_equal( mm[[1]]$w[21], 0.04986543, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal )
    expect_relopt( mm, ff )

    ## Test bias fixture
    mm[[5]] <- ftrain( params[[4]], fix_bias=TRUE )
    expect_equal( mm[[5]]$b, with(params[[4]], sum(a*z)/sum(a)) )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[5]]) )

    ## Test non-negativity
    mm[[6]] <- ftrain( params[[4]], nonneg=TRUE )
    purrr::map( mm[[6]]$w, expect_gte, 0 )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[6]]) )

    ## Compose model definitions using the "grammar of modeling"
    dd <- gen_modeldef_lin( params )

    ## Train based on model definitions
    ## Ensure equivalence to direct calling of gelnet_lin_opt()
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )

    ## Test the L1 ceiling computation
    l1c <- with( params[[4]], l1c_lin( X, z, l2, a, d, P, m ) )
    m1 <- ftrain(params[[4]], l1=l1c)
    m2 <- ftrain(params[[4]], l1=l1c-0.01)
    expect_equal( sum(m1$w), 0 )
    expect_length( which(m2$w != 0), 1 )
})

