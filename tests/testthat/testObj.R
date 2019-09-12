context("Objective functions")

test_that( "Linear regression objective", {
    ## Load pre-generated data
    load( "data/lin.RData" )
    
    ## Generate random model weights and bias
    set.seed(100)
    w <- rnorm( ncol(params$X) )
    b <- rnorm( 1 )

    ## Evaluate increasingly complex models
    f1 <- with( params, gelnet_lin_obj(w, b, X, z, 0, 0.1) )
    f2 <- with( params, gelnet_lin_obj(w, b, X, z, 0.1, 0.1) )
    f3 <- with( params, gelnet_lin_obj(w, b, X, z, 0.1, 0.1, a, d) )
    f4 <- with( params, gelnet_lin_obj(w, b, X, z, 0.1, 0.1, a, d, P) )
    f5 <- with( params, gelnet_lin_obj(w, b, X, z, 0.1, 0.1, a, d, P, m) )

    ## Compare to fixed values
    expect_equal( f1, 18.26007, tol=1e-5 )
    expect_equal( f2, 21.33668, tol=1e-5 )
    expect_equal( f3, 7.397314, tol=1e-5 )
    expect_equal( f4, 6.800982, tol=1e-5 )
    expect_equal( f5, 6.758627, tol=1e-5 )
})

test_that( "Binary logistic regression objective", {
    ## Load pre-generated data
    load( "data/blr.RData" )
    
    ## Generate random model weights and bias
    set.seed(100)
    w <- rnorm( ncol(params$X) )
    b <- rnorm( 1 )

    ## Evaluate increasingly complex models
    f1 <- with( params, gelnet_blr_obj(w, b, X, y, 0, 0.01) )
    f2 <- with( params, gelnet_blr_obj(w, b, X, y, 0.1, 0.1) )
    f3 <- with( params, gelnet_blr_obj(w, b, X, y, 0.1, 0.1, FALSE, d) )
    f4 <- with( params, gelnet_blr_obj(w, b, X, y, 0.1, 0.1, FALSE, d, P) )
    f5 <- with( params, gelnet_blr_obj(w, b, X, y, 0.1, 0.1, FALSE, d, P, m) )
    f6 <- with( params, gelnet_blr_obj(w, b, X, y, 0.1, 0.1, TRUE, d, P, m) )
    
    ## Compare to fixed values
    expect_equal( f1, 4.011542, tol=1e-5 )
    expect_equal( f2, 8.582172, tol=1e-5 )
    expect_equal( f3, 6.902192, tol=1e-5 )
    expect_equal( f4, 6.305861, tol=1e-5 )
    expect_equal( f5, 6.250091, tol=1e-5 )
    expect_equal( f6, 6.333796, tol=1e-5 )
})

test_that( "One-class logistic regression objective", {
    ## Load pre-generated data
    load( "data/oclr.RData" )
    
    ## Generate random model weights
    set.seed(100)
    w <- rnorm( ncol(params$X) )
    
    ## Evaluate increasingly complex models
    f1 <- with( params, gelnet_oclr_obj(w, X, 0, 0.01) )
    f2 <- with( params, gelnet_oclr_obj(w, X, 0.1, 0.1) )
    f3 <- with( params, gelnet_oclr_obj(w, X, 0.1, 0.1, d) )
    f4 <- with( params, gelnet_oclr_obj(w, X, 0.1, 0.1, d, P) )
    f5 <- with( params, gelnet_oclr_obj(w, X, 0.1, 0.1, d, P, m) )
    
    ## Compare to fixed values
    expect_equal( f1, 1.920498, tol=1e-5 )
    expect_equal( f2, 6.491128, tol=1e-5 )
    expect_equal( f3, 4.775318, tol=1e-5 )
    expect_equal( f4, 4.178987, tol=1e-5 )
    expect_equal( f5, 4.230996, tol=1e-5 )
})
