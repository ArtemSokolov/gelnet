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

test_that( "linear training", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Preset L1 and L2 penalty coefficients
    L1 <- 0.1
    L2 <- 10

    ## Generate random data
    set.seed(100)
    X <- matrix( rnorm(n*p), n, p )
    z <- rnorm( n )

    ## Generate random sample and feature weights
    a <- runif( n )
    d <- runif( p )

    ## Generate random feature-feature penalties and translation coeffs
    A <- matrix( rnorm(p*p), p, p )
    P <- t(A) %*% A / p
    m <- rnorm(p, sd= 0.1)

    ## Train a basic model
    m1 <- gelnet.lin( X, z, L1, L2, silent=TRUE )
    f1 <- function( mdl ) { gelnet.lin.obj( mdl$w, mdl$b, X, z, L1, L2 ) }

    ## Verify the basic model
    expect_optimal( m1, f1 )
    expect_length( which( m1$w != 0 ), 25 )
    expect_equal( m1$b, 0.110697, tol=1e-5 )
})

