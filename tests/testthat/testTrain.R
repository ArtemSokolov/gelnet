context( "Model training" )

## Ensures that training found the optimum by looking in the immediate
##  neighborhood of the solution model
## model - a model object, as returned by, e.g., gelnet.lin()
## fobj - the matching objective function to minimize
expect_optimal <- function( model, fobj )
{
    ## Predefine the error message
    msg <- "Model is not optimized along component "

    ## Capture the actual value and its label in the parent env
    act <- quasi_label( rlang::enquo(model) )

    ## Compute the objective function value of the trained model
    act$p <- length(act$val$w)
    act$obj <- fobj( act$val )

    ## Consider the immediate neighborhood of the solution
    for( i in 0:act$p )
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

test_that( "linear training", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Silently trains a linear GELnet model using the provided parameters
    ftrain <- function( prms )
        do.call( gelnet.lin, c(prms, list(silent=TRUE)) )

    ## Generates a model evaluator using a given set of parameters
    fgen <- function( prms )
        { function( mdl ) { do.call( gelnet.lin.obj, c(mdl,prms) ) } }

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
})

