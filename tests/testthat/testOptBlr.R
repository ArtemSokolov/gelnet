context( "Binary logistic regression model training" )

source( "custom.R" )

test_that( "Binary logistic regression training", {
    load( "data/blr.RData" )

    ## Consider models of increasing complexity by subsampling
    ##   the parameter space
    vv <- purrr::accumulate( list(c("l1", "l2", "y", "X"),
                                  "d", "P", "m"), c )

    ## Define training functions and compute corresponding models
    ftrain <- purrr::map( vv, ~partial2(gelnet_blr_opt, params[.x]) )
    mm <- purrr::map( ftrain, do.call, list(silent=TRUE) )

    ## Generate the models and matching objective functions
    fobj <- purrr::map( vv, ~partial2(gelnet_blr_obj, params[.x]) )
    ff <- purrr::map( fobj, purrr::lift_dl )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 21 )
    expect_equal( mm[[1]]$b, 0.4139836, tol=1e-5 )
    expect_equal( mm[[1]]$w[21], 0.02048664, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal )
    expect_relopt( mm, ff )

    ## Compose model definitions using the "grammar of modeling"
    dd <- list()
    dd[[1]] <- with( params, gelnet(X) + model_blr(factor(y, c(1,0))) +
                             rglz_L1(l1) + rglz_L2(l2) )
    dd[[2]] <- dd[[1]] + rglz_L1( params$l1, params$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params$l2, params$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params$l2, params$P, params$m )

    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})

test_that( "Handling class imbalance", {
    load( "data/blr.RData" )

    ## Define the training and objective function
    ftrain <- partial2( gelnet_blr_opt, params, silent=TRUE )
    fobj <- purrr::lift_dl( partial2(gelnet_blr_obj, params) )

    ## Compute models with and without balance
    m1 <- ftrain()
    m2 <- ftrain( balanced=TRUE )

    ## Balanced model should have bias term closer to 0
    expect_lt( abs(m2$b), abs(m1$b) )

    ## Compare the two models against the objective function
    expect_lt( fobj(m1), fobj(m2) )

    ## Verify that grammar of modeling produces the same result
    mdef <- with( params, gelnet(X) + model_blr(factor(y, c(1,0)), balanced=TRUE) +
                          rglz_L1(l1, d) + rglz_L2(l2, P, m) )
    expect_equal( gelnet_train(mdef, silent=TRUE), m2 )
})

test_that( "Non-negativity", {
    load( "data/blr.RData" )

    ## Define the training and objective function
    ftrain <- partial2( gelnet_blr_opt, params, silent=TRUE )
    fobj <- purrr::lift_dl( partial2(gelnet_blr_obj, params) )

    ## Compute models with and without enforced negativity
    m1 <- ftrain()
    m2 <- ftrain( nonneg=TRUE )

    ## Ensure non-negativity constraint is satisfied
    purrr::map( m2$w, expect_gte, 0 )

    ## Compare the two models against the objective function
    expect_lt( fobj(m1), fobj(m2) )

    ## Verify that grammar of modeling produces the same result
    mdef <- with( params, gelnet(X) + model_blr(factor(y, c(1,0)), nonneg=TRUE) +
                          rglz_L1(l1, d) + rglz_L2(l2, P, m) )
    expect_equal( gelnet_train(mdef, silent=TRUE), m2 )
})

