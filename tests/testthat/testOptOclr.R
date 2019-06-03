context( "One-class logistic regression model training" )

source( "custom.R" )

test_that( "One-class logistic regression training", {
    load( "data/oclr.RData" )

    ## Consider models of increasing complexity by subsampling
    ##   the parameter space
    vv <- purrr::accumulate( list(c("l1", "l2", "X"), "d", "P", "m"), c )

    ## Define training functions and compute corresponding models
    ftrain <- purrr::map( vv, ~partial2(gelnet_oclr_opt, params[.x]) )
    mm <- purrr::map( ftrain, do.call, list(silent=TRUE) )

    ## Generate the models and matching objective functions
    fobj <- purrr::map( vv, ~partial2(gelnet_oclr_obj, params[.x]) )
    ff <- purrr::map( fobj, purrr::lift_dl )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 17 )
    expect_equal( mm[[1]]$w[26], -0.01308364, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal, FALSE )
    expect_relopt( mm, ff )

    ## Compose model definitions using the "grammar of modeling"
    dd <- list()
    dd[[1]] <- with( params, gelnet(X) + rglz_L1(l1) + rglz_L2(l2) )
    dd[[2]] <- dd[[1]] + rglz_L1( params$l1, params$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params$l2, params$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params$l2, params$P, params$m )

    ## Train based on model definitions
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )
})

test_that( "Non-negativity", {
    load( "data/oclr.RData" )

    ## Define the training and objective function
    ftrain <- partial2( gelnet_oclr_opt, params, silent=TRUE )
    fobj <- purrr::lift_dl( partial2(gelnet_oclr_obj, params) )

    ## Compute models with and without enforced negativity
    m1 <- ftrain()
    m2 <- ftrain( nonneg=TRUE )

    ## Ensure non-negativity constraint is satisfied
    purrr::map( m2$w, expect_gte, 0 )

    ## Compare the two models against the objective function
    expect_lt( fobj(m1), fobj(m2) )

    ## Verify that grammar of modeling produces the same result
    mdef <- with( params, gelnet(X) + rglz_L1(l1,d) +
                          rglz_L2(l2,P,m) + model_oclr(TRUE) )
    expect_equal( gelnet_train(mdef, silent=TRUE), m2 )
})

