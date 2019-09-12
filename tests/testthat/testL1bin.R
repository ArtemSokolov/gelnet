context( "Binary search" )

## Wrapper for testing binary search
test_l1bin <- function( mdf, d=NULL, nf=c(1,5,10,25,50) )
{
    mdfs <- purrr::map( nf, ~(mdf + rglz_nf(.x,d)) )
    mdls <- purrr::map( mdfs, gelnet_train, silent=TRUE )
    wnz <- purrr::map( mdls, ~which(.x$w != 0) )
    purrr::map2( wnz, nf, expect_length )
}

test_that( "Linear regression", {
    load( "data/lin.RData" )

    mdf <- list()
    mdf[[1]] <- with(params, gelnet(X) + model_lin(z,a) + rglz_L2(l2,P,m))
    mdf[[2]] <- with(params, gelnet(X) + model_lin(z,a,fix_bias=TRUE) + rglz_L2(l2,P,m))
    purrr::map( mdf, test_l1bin, params$d )

    ## Maxes out at 23 non-negative features
    mdf_nn <- with(params, gelnet(X) + model_lin(z,a,nonneg=TRUE) + rglz_L2(l2,P,m))
    test_l1bin( mdf_nn, params$d, c(1,5,10,23) )

    ## Check for failure at 24 features
    expect_error(test_l1bin( mdf_nn, params$d, 24 ), "Unable to reach")
})

test_that( "Binary logistic regression", {
    load( "data/blr.RData" )

    mdf <- list()
    yf <- factor(params$y, c(1,0))
    mdf[[1]] <- with(params, gelnet(X) + model_blr(yf) + rglz_L2(l2,P,m))
    mdf[[2]] <- with(params, gelnet(X) + model_blr(yf, balanced=TRUE) + rglz_L2(l2,P,m))
    purrr::map( mdf, test_l1bin, params$d )

    ## Maxes out at 25 non-negative features
    mdf_nn <- with(params, gelnet(X) + model_blr(yf, nonneg=TRUE) + rglz_L2(l2,P,m))
    test_l1bin( mdf_nn, params$d, c(1,5,10,25) )

    ## Check for failure at 26 features
    expect_error( test_l1bin(mdf_nn, params$d, 26), "Unable to reach" )
})

test_that( "One-class logistic regression", {
    load( "data/oclr.RData" )

    mdf <- with( params, gelnet(X) + rglz_L2(l2,P,m) )
    test_l1bin( mdf, params$d )

    ## Maxes out at 28 non-negative features
    mdf_nn <- with(params, gelnet(X) + model_oclr(TRUE) + rglz_L2(l2,P,m))
    test_l1bin( mdf_nn, params$d, c(1,5,10,28) )

    ## Check for failure at 29 features
    expect_error( test_l1bin(mdf_nn, params$d, 29), "Unable to reach" )
})
