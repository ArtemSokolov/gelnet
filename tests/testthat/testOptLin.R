context( "Linear regression model training" )

source( "custom.R" )

## Returns increasingly complex instantiations of f based on pre-generated
##   linear regression data
partial_lin <- function(f)
{
    load( "data/lin.RData" )
    vv <- purrr::accumulate( list(c("l1", "l2", "z", "X"),
                                  c("a", "d"), "P", "m"), c )
    pp <- purrr::map( vv, ~c(list(f), params[.x]) )
    purrr::map( pp, ~do.call(purrr::partial, .x) )
}

## Generates model definitions based on the provided set of parameters
gen_modeldef_lin <- function()
{
    load( "data/lin.RData" )
    dd <- list()
    dd[[1]] <- gelnet( params$X ) + model_lin( params$z ) +
        rglz_L1( params$l1 ) + rglz_L2( params$l2 )
    dd[[2]] <- dd[[1]] + model_lin( params$z, params$a ) +
        rglz_L1( params$l1, params$d )
    dd[[3]] <- dd[[2]] + rglz_L2( params$l2, params$P )
    dd[[4]] <- dd[[3]] + rglz_L2( params$l2, params$P, params$m )
    dd[[5]] <- dd[[4]] + model_lin( params$z, params$a, fix_bias=TRUE )
    dd[[6]] <- dd[[4]] + model_lin( params$z, params$a, nonneg=TRUE )
    dd
}

test_that( "Linear regression training", {
    ftrain <- partial_lin( gelnet_lin_opt )
    fobj <- partial_lin( gelnet_lin_obj )

    mm <- purrr::map( ftrain, do.call, list(silent=TRUE) )
    ff <- purrr::map( fobj, purrr::lift_dl )

    ## Verify the basic model
    expect_length( which( mm[[1]]$w != 0 ), 30 )
    expect_equal( mm[[1]]$b, 0.06710631, tol=1e-5 )
    expect_equal( mm[[1]]$w[21], 0.04986543, tol=1e-5 )

    ## Verify optimality of each model w.r.t. its obj. fun.
    purrr::map2( mm, ff, expect_optimal )
    expect_relopt( mm, ff )

    ## Test bias fixture
    load( "data/lin.RData" )
    mm[[5]] <- ftrain[[4]]( silent=TRUE, fix_bias=TRUE )
    expect_equal( mm[[5]]$b, with(params, sum(a*z)/sum(a)) )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[5]]) )

    ## Test non-negativity
    mm[[6]] <- ftrain[[4]]( silent=TRUE, nonneg=TRUE )
    purrr::map( mm[[6]]$w, expect_gte, 0 )
    expect_lt( ff[[4]](mm[[4]]), ff[[4]](mm[[6]]) )

    ## Compose model definitions using the "grammar of modeling"
    dd <- gen_modeldef_lin()

    ## Train based on model definitions
    ## Ensure equivalence to direct calling of gelnet_lin_opt()
    mdls <- purrr::map( dd, gelnet_train, silent=TRUE )
    purrr::map2( mm, mdls, expect_equal )

    ## Test the L1 ceiling computation
    l1c <- with( params, l1c_lin( X, z, l2, a, d, P, m ) )
    m1 <- gen_ftrain( gelnet_lin_opt )( params, l1=l1c-0.0001)
    m2 <- gen_ftrain( gelnet_lin_opt )( params, l1=l1c+0.0001)
    expect_length( which(m1$w != 0), 1 )
    expect_equal( sum(m2$w), 0 )
})

