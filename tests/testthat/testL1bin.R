context( "Binary search" )

test_that( "Linear regression", {
    load( "data/lin.RData" )

    v <- c(1,5,10,25,50)
    mdfs <- purrr::map( v, ~with(params, gelnet(X)+model_lin(z,a)+rglz_nf(.x,d)+rglz_L2(l2,P,m)) )
    mdls <- purrr::map( mdfs, gelnet_train, silent=TRUE )
    nf <- purrr::map( mdls, ~which(.x$w != 0) )
    purrr::map2( nf, v, expect_length )
})
