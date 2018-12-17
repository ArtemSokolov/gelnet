context( "Model training" )

## Ensures that training found the optimum by looking in the immediate
##  neighborhood of the solution model
## model - a model object, as returned by, e.g., gelnet_lin_opt()
## fobj - the matching objective function to minimize
## hasBias - set to TRUE if the model has a bias term that needs to be tested
expect_optimal <- function( model, fobj, hasBias = TRUE )
{
    ## Predefine the error message
    msg <- "Model is not optimized along component "

    ## Capture the actual value and its label in the parent env
    act <- quasi_label( rlang::enquo(model) )

    ## Compute the objective function value of the trained model
    act$p <- length(act$val$w)
    act$obj <- fobj( act$val )

    ## Consider the immediate neighborhood of the solution
    iStart <- as.integer( !hasBias )
    for( i in iStart:act$p )
    {
        act$objn <- fobj( perturb.gelnet(act$val, i, -0.001) )
        act$objp <- fobj( perturb.gelnet(act$val, i, 0.001) )
        expect( act$objp > act$obj, stringr::str_c(msg, i) )
        expect( act$objn > act$obj, stringr::str_c(msg, i) )
    }

    ## Return the actual value (invisibly for pipe usage)
    invisible(act$val)
}

## A generalization of expect_optimal() that compares model optimality
##  in the context of other models / objective functions
## lmd - list of models
## lfn - list of matching function objectives
expect_relopt <- function( lmd, lfn )
{
    ## Capture the actual value and its label in the parent env
    act <- quasi_label( rlang::enquo(lmd) )

    ## Verify the arguments
    expect( length(lmd) == length(lfn),
           str_c("Each model in ", act$lab, " must have a matching objective function") )

    ## Traverse the models. For each, ensure it is more optimal for the
    ## corresponding objective function than other models in the list
    for( i in 1:length(lmd) )
    {
        ## Apply the i^th objective function to each model
        v <- purrr::map( lmd, lfn[[i]] )

        ## Ensure that the i^th model produces the lowest object value
        expect( which.min( v ) == i,
               glue::glue( "Model {i} in {act$lab} is outperformed on its ",
                          "objective function value" ) )
    }

    ## Return the actual value (invisibly for pipe usage)
    invisible(act$val)
}

test_that( "Linear regression training", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Silently trains a linear GELnet model using the provided parameters
    ftrain <- function( prms )
        do.call( gelnet_lin_opt, c(prms, list(silent=TRUE)) )

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
        { function( mdl ) { do.call( gelnet_lin_obj, c(mdl,prms) ) } }

    ## Preset L1 and L2 penalty coefficients and generate a sequence of
    ##   models increasing in complexity
    set.seed(100)
    A <- matrix( rnorm(p*p), p, p )
    params <- list( list(l1 = 0.1, l2 = 10, z = rnorm(n),
                         X = matrix(rnorm(n*p), n, p)) )
    params[[2]] <- c( params[[1]], list(a = runif( n ), d = runif( p )) )
    params[[3]] <- c( params[[2]], list( P = t(A) %*% A / p ) )
    params[[4]] <- c( params[[3]], list(m = rnorm(p, sd=0.1)) )

    ## Generate the models and matching objective functions
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
    mm[[5]] <- do.call( gelnet_lin_opt, c(params[[4]], list(silent=TRUE, fix_bias=TRUE)) )
    expect_equal( mm[[5]]$b, with(params[[4]], sum(a*z)/sum(a)) )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[5]]) )

    ## Test non-negativity
    mm[[6]] <- do.call( gelnet_lin_opt, c(params[[4]], list(silent=TRUE, nonneg=TRUE)) )
    purrr::map( mm[[6]]$w, expect_gte, 0 )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[6]]) )

    ## Compose model definitions using the "grammar of modeling"
    dd <- list()
    dd[[1]] <- gelnet( params[[1]]$X ) + model_lin( params[[1]]$z ) +
        rglz_L1( params[[1]]$l1 ) + rglz_L2( params[[1]]$l2 )
    dd[[2]] <- dd[[1]] + model_lin( params[[2]]$z, params[[2]]$a ) +
        rglz_L1( params[[2]]$l1, params[[2]]$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params[[3]]$l2, params[[3]]$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params[[4]]$l2, params[[4]]$P, params[[4]]$m )
    dd[[5]] <- dd[[4]] + model_lin( params[[4]]$z, params[[4]]$a, fix_bias=TRUE )
    dd[[6]] <- dd[[4]] + model_lin( params[[4]]$z, params[[4]]$a, nonneg=TRUE )

    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})

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
    mnn <- do.call( gelnet_blr_opt, c(params[[5]], list(nonneg=TRUE, silent=TRUE)) )
    purrr::map( mnn$w, expect_gte, 0 )
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
    mnn <- do.call( gelnet_oclr_opt, c(params[[4]], list(silent=TRUE, nonneg=TRUE)) )
    purrr::map( mnn$w, expect_gte, 0 )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mnn) )
})
