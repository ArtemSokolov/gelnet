context("Objective functions")

test_that( "linear objective", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Generate random model weights, bias and random data
    set.seed(100)
    w <- rnorm( p )
    b <- rnorm( 1 )
    X <- matrix( rnorm(n*p), n, p )
    z <- rnorm( n )

    ## Generate random sample and feature weights
    a <- runif( n )
    d <- runif( p )

    ## Generate random feature-feature penalties and translation coeffs
    A <- matrix( rnorm(p*p), p, p )
    P <- t(A) %*% A / p
    m <- rnorm(p, sd= 0.1)

    ## Evaluate increasingly complex models
    expect_equal( rcpp_gelnet_lin_obj( w, b, X, z, 0, 0.1 ), 15.81988, tol=1e-5 )
    expect_equal( rcpp_gelnet_lin_obj( w, b, X, z, 0.1, 0.1 ), 18.8965, tol=1e-5 )
    expect_equal( rcpp_gelnet_lin_obj( w, b, X, z, 0.1, 0.1, a, d ), 9.982413, tol=1e-5 )
    expect_equal( rcpp_gelnet_lin_obj( w, b, X, z, 0.1, 0.1, a, d, P ), 10.11774, tol=1e-5 )
    expect_equal( rcpp_gelnet_lin_obj( w, b, X, z, 0.1, 0.1, a, d, P, m ), 10.09573, tol=1e-5 )
})

test_that( "Logistic regression objective", {
    ## Preset dimensionality
    n <- 20
    p <- 50

    ## Generate random model weights, bias and random data
    set.seed(100)
    w <- rnorm( p )
    b <- rnorm( 1 )
    X <- matrix( rnorm(n*p), n, p )
    y <- sample( c(0,1), n, replace=TRUE )

    ## Generate random sample and feature weights, feature-feature
    ##  penalties and translation coeffs
    d <- runif( p )
    A <- matrix( rnorm(p*p), p, p )
    P <- t(A) %*% A / p
    m <- rnorm(p, sd= 0.1)

    ## Evaluate increasingly complex models
    expect_equal( rcpp_gelnet_logreg_obj( w, b, X, y, 0, 0.01 ), 1.409101, tol=1e-5 )
    expect_equal( rcpp_gelnet_logreg_obj( w, b, X, y, 0.1, 0.1 ), 5.979731, tol=1e-5 )
    expect_equal( rcpp_gelnet_logreg_obj( w, b, X, y, 0.1, 0.1, FALSE, d ), 4.673295, tol=1e-5 )
    expect_equal( rcpp_gelnet_logreg_obj( w, b, X, y, 0.1, 0.1, FALSE, d, P ),
                 4.952671, tol=1e-5 )
    expect_equal( rcpp_gelnet_logreg_obj( w, b, X, y, 0.1, 0.1, FALSE, d, P, m ),
                 4.960332, tol=1e-5 )
})
