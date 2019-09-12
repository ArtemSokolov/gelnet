context( "Linear regression model training" )

source( "custom.R" )

test_that( "Linear regression training", {
    load( "data/lin.RData" )

    ## Consider models of increasing complexity by subsampling
    ##   the parameter space
    vv <- purrr::accumulate( list(c("l1", "l2", "z", "X"),
                                  c("a", "d"), "P", "m"), c )

    ## Define training functions and compute corresponding models
    ftrain <- purrr::map( vv, ~partial2(gelnet_lin_opt, params[.x]) )
    mm <- purrr::map( ftrain, do.call, list(silent=TRUE) )

    ## Generate matching objective functions
    fobj <- purrr::map( vv, ~partial2(gelnet_lin_obj, params[.x]) )
    ff <- purrr::map( fobj, purrr::lift_dl )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 30 )
    expect_equal( mm[[1]]$b, 0.06710631, tol=1e-5 )
    expect_equal( mm[[1]]$w[21], 0.04986543, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal )
    expect_relopt( mm, ff )

    ## Compose model definitions using the "grammar of modeling"
    dd <- list()
    dd[[1]] <- with( params, gelnet(X) + model_lin(z) + rglz_L1(l1) + rglz_L2(l2) )
    dd[[2]] <- with( params, dd[[1]] + model_lin(z, a) + rglz_L1(l1, d) )
    dd[[3]] <- with( params, dd[[2]] + rglz_L2(l2, P) )
    dd[[4]] <- with( params, dd[[3]] + rglz_L2(l2, P, m) )

    ## Train based on model definitions
    ## Ensure equivalence to direct calling of gelnet_lin_opt()
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})

test_that( "Fixing bias term", {
    load( "data/lin.RData" )

    ## Define the training and objective function
    ftrain <- partial2( gelnet_lin_opt, params, silent=TRUE )
    fobj <- purrr::lift_dl( partial2(gelnet_lin_obj, params) )

    ## Compute models with and without a fixed bias term
    m1 <- ftrain()
    m2 <- ftrain( fix_bias=TRUE )

    ## Verify the value of the bias term
    expect_equal( m2$b, with(params, sum(a*z)/sum(a)) )

    ## Compare models with and without a fixed bias term
    ##   w.r.t. the objective function
    expect_lt( fobj(m1), fobj(m2) )

    ## Verify that grammar of modeling produces the same result
    mdef <- with( params, gelnet(X) + model_lin(z, a, fix_bias=TRUE) +
                          rglz_L1(l1, d) + rglz_L2(l2, P, m) )
    expect_equal( gelnet_train(mdef, silent=TRUE), m2 )
})

test_that( "Non-negativity", {
    load( "data/lin.RData" )

    ## Define the training and objective function
    ftrain <- partial2( gelnet_lin_opt, params, silent=TRUE )
    fobj <- purrr::lift_dl( partial2(gelnet_lin_obj, params) )

    ## Compute models with and without enforced negativity
    m1 <- ftrain()
    m2 <- ftrain( nonneg=TRUE )

    ## Ensure non-negativity constraint is satisfied
    purrr::map( m2$w, expect_gte, 0 )

    ## Compare models with and without enforced negativity
    ##   w.r.t the objective function
    expect_lt( fobj(m1), fobj(m2) )

    ## Verify that grammar of modeling produces the same result
    mdef <- with( params, gelnet(X) + model_lin(z, a, nonneg=TRUE) +
                          rglz_L1(l1, d) + rglz_L2(l2, P, m) )
    expect_equal( gelnet_train(mdef, silent=TRUE), m2 )
})

