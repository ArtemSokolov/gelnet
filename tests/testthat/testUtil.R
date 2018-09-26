context( "Utility functions" )

test_that( "Model Perturbations", {
    ## Compose a random model
    set.seed(100)
    mdl <- list( w = rnorm(10), b = rnorm(1) )

    ## Generate two perturbations (one for bias, one for weights)
    mdl1 <- perturb.gelnet( mdl, 0, 0.01 )
    mdl2 <- perturb.gelnet( mdl, 5, -0.01 )

    ## Evalute the first perturbation
    expect_equal( mdl$w, mdl1$w )
    expect_equal( mdl1$b - mdl$b, 0.01 )

    ## Evaluate the second perturbation
    expect_equal( mdl2$w[5] - mdl$w[5], -0.01 )
    expect_equal( mdl2$w[-5], mdl$w[-5] )
    expect_equal( mdl2$b, mdl$b )
})
