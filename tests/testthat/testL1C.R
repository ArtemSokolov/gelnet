context( "L1 ceiling" )

source( "custom.R" )

## Wrapper for testing L1 ceiling values
## ftrain - model training function
## params - set of parameters to supply
## l1c - ceiling value to test
test_l1c <- function( ftrain, params, l1c )
{
    m1 <- partial2( ftrain, params, l1=l1c-0.0001 )(silent=TRUE)
    m2 <- partial2( ftrain, params, l1=l1c+0.0001 )(silent=TRUE)
    expect_length( which(m1$w != 0), 1 )
    expect_equal( sum(m2$w), 0 )
}

test_that( "Linear regression", {
    load( "data/lin.RData" )

    ## L1 ceiling for linear regression
    l1c <- with( params, l1c_lin( X, z, l2, a, d, P, m ) )
    test_l1c( gelnet_lin_opt, params, l1c )

    ## Ensure the "grammar of modeling" comes up with the same
    ##   L1 ceiling value
    mdef <- with( params, gelnet(X) + model_lin(z,a) +
                          rglz_L1(l1,d) + rglz_L2(l2,P,m) )
    expect_equal( L1_ceiling(mdef), l1c )
})

test_that( "Binary logistic regression", {
    load( "data/blr.RData" )

    ## L1 ceiling for binary logistic regression (unbalanced case)
    l1c1 <- with( params, l1c_blr(X, y, l2, FALSE, d, P, m) )
    ftrain1 <- purrr::partial( gelnet_blr_opt, eps=1e-20 )
    test_l1c( ftrain1, params, l1c1 )

    ## L1 ceiling for binary logistic regression (balanced case)
    l1c2 <- with( params, l1c_blr(X, y, l2, TRUE, d, P, m) )
    ftrain2 <- purrr::partial( gelnet_blr_opt, eps=1e-20, balanced=TRUE )
    test_l1c( ftrain2, params, l1c2 )

    ## Ensure the "grammar of modeling" comes up with the same
    ##   L1 ceiling values
    mdef1 <- with( params, gelnet(X) + model_blr(factor(y, c(1,0))) +
                           rglz_L1(l1,d) + rglz_L2(l2,P,m) )
    mdef2 <- with( params, gelnet(X) + model_blr(factor(y, c(1,0)), balanced=TRUE) +
                           rglz_L1(l1,d) + rglz_L2(l2,P,m) )
    expect_equal( L1_ceiling(mdef1), l1c1 )
    expect_equal( L1_ceiling(mdef2), l1c2 )
})

test_that( "One-class logistic regression", {
    load( "data/oclr.RData" )

    ## L1 ceiling for one-class logistic regression
    l1c <- with( params, l1c_oclr(X, l2, d, P, m) )
    test_l1c( gelnet_oclr_opt, params, l1c )

    ## Ensure the "grammar of modeling" comes up with the same
    ##   L1 ceiling value
    mdef <- with( params, gelnet(X) + rglz_L1(l1,d) + rglz_L2(l2,P,m) )
    expect_equal( L1_ceiling(mdef), l1c )
})
