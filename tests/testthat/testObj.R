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
    expect_equal( gelnet.lin.obj( w, b, X, z, 0, 0.1 ), matrix(15.81988), tol=1e-5 )
    expect_equal( gelnet.lin.obj( w, b, X, z, 0.1, 0.1 ), matrix(18.8965), tol=1e-5 )
    expect_equal( gelnet.lin.obj( w, b, X, z, 0.1, 0.1, a, d ), matrix(9.982413), tol=1e-5 )
    expect_equal( gelnet.lin.obj( w, b, X, z, 0.1, 0.1, a, d, P ), matrix(10.11774), tol=1e-5 )
    expect_equal( gelnet.lin.obj( w, b, X, z, 0.1, 0.1, a, d, P, m ), matrix(10.09573), tol=1e-5 )
})

