context( "Model training" )

source( "custom.R" )

test_that( "Binary logistic regression training", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Silently trains a logistic GELnet model using the provided parameters
    ftrain <- function( prms )
    { do.call( gelnet_blr_opt, c(prms, list(silent=TRUE)) ) }

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
    { function( mdl ) { do.call( gelnet_blr_obj, c(mdl,prms) ) } }

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, y = sample( c(0,1), n, replace=TRUE ),
                         X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- c( params[[1]], list(d = runif( p )) )
    params[[3]] <- c( params[[2]], list(P = t(A) %*% A / p) )
    params[[4]] <- c( params[[3]], list(m = rnorm(p, sd=0.1)) )
    params[[5]] <- c( params[[4]], list(balanced=TRUE) )

    ## Generate the models and matching objective functions
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
    mm[[6]] <- do.call( gelnet_blr_opt, c(params[[5]], list(nonneg=TRUE, silent=TRUE)) )
    purrr::map( mm[[6]]$w, expect_gte, 0 )
    expect_lt( ff[[5]](mm[[5]]), ff[[5]](mm[[6]]) )
    
    ## Compose model definitions using the "grammar of modeling"
    dd <- list()
    dd[[1]] <- gelnet(params[[1]]$X) + model_blr(factor(params[[1]]$y, c(1,0))) +
        rglz_L1( params[[1]]$l1 ) + rglz_L2( params[[1]]$l2 )
    dd[[2]] <- dd[[1]] + rglz_L1( params[[2]]$l1, params[[2]]$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params[[3]]$l2, params[[3]]$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params[[4]]$l2, params[[4]]$P, params[[4]]$m )
    dd[[5]] <- dd[[4]] + model_blr( factor(params[[5]]$y, c(1,0)), balanced=TRUE )
    dd[[6]] <- dd[[5]] + model_blr( factor(params[[5]]$y, c(1,0)), TRUE, TRUE )

    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})

test_that( "One-class logistic regression training", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Silently trains a logistic GELnet model using the provided parameters
    ftrain <- function( prms )
    { do.call( gelnet_oclr_opt, c(prms, list(silent=TRUE)) ) }

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
    { function( mdl ) { do.call( gelnet_oclr_obj, c(mdl,prms) ) } }

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- c( params[[1]], list(d = runif( p )) )
    params[[3]] <- c( params[[2]], list(P = t(A) %*% A / p) )
    params[[4]] <- c( params[[3]], list(m = rnorm(p, sd=0.1)) )

    ## Generate the models and matching objective functions
    mm <- purrr::map( params, ftrain )
    ff <- purrr::map( params, fgen )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 17 )
    expect_equal( mm[[1]]$w[26], -0.01308364, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal, FALSE )
    expect_relopt( mm, ff )

    ## Test non-negativity
    mm[[5]] <- do.call( gelnet_oclr_opt, c(params[[4]], list(silent=TRUE, nonneg=TRUE)) )
    purrr::map( mm[[5]]$w, expect_gte, 0 )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[5]]) )

    ## Compose model definitions using the "grammar of modeling"
    dd <- list()
    dd[[1]] <- gelnet( params[[1]]$X ) + rglz_L1( params[[1]]$l1 ) + rglz_L2( params[[2]]$l2 )
    dd[[2]] <- dd[[1]] + rglz_L1( params[[2]]$l1, params[[2]]$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params[[3]]$l2, params[[3]]$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params[[4]]$l2, params[[4]]$P, params[[4]]$m )
    dd[[5]] <- dd[[4]] + model_oclr(TRUE)

    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})
