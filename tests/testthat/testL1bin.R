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

    ## Check for failure at 25 features
    ## TODO
    test_l1bin( mdf_nn, params$d, 25 )
})

## test_that( "Binary logistic regression", {
##     load( "data/blr.RData" )

    
## })
