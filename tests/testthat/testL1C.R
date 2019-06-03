context( "L1 ceiling" )

## Generates a silent model training function
## prms - preset parameters
## ... - dynamic parameters
gen_ftrain <- function( f )
{
    function( prms, ... )
    {
        p <- purrr::list_modify( prms, silent=TRUE, ... )
        do.call( f, p )
    }
}

test_that( "Linear regression", {
    load( "data/lin.RData" )

    ## Test the L1 ceiling computation
    l1c <- with( params, l1c_lin( X, z, l2, a, d, P, m ) )
    m1 <- gen_ftrain( gelnet_lin_opt )( params, l1=l1c-0.0001)
    m2 <- gen_ftrain( gelnet_lin_opt )( params, l1=l1c+0.0001)
    expect_length( which(m1$w != 0), 1 )
    expect_equal( sum(m2$w), 0 )
})

test_that( "Binary logistic regression", {
    load( "data/blr.RData" )

    ## Set up the training function
    ftrain <- gen_ftrain( gelnet_blr_opt )

    ## Test the L1 ceiling computation (unbalanced case)
    l1c <- with( params, l1c_blr(X, y, l2, FALSE, d, P, m) )
    m1 <- ftrain( params, l1=l1c-0.0001, eps=1e-20 )
    m2 <- ftrain( params, l1=l1c+0.0001, eps=1e-20 )
    expect_length( which(m1$w != 0), 1 )
    expect_equal( sum(m2$w), 0 )

    ## Test the L1 ceiling computation (balanced case)
    l1cb <- with( params, l1c_blr(X, y, l2, TRUE, d, P, m) )
    m1b <- ftrain( params, l1=l1cb-0.6, eps=1e-20 )
    m2b <- ftrain( params, l1=l1cb+0.0001, eps=1e-20 )
    expect_length( which(m1b$w != 0), 1 )
    expect_equal( sum(m2b$w), 0 )
})

test_that( "One-class logistic regression", {
    load( "data/oclr.RData" )

    ## Set up the training function
    ftrain <- gen_ftrain( gelnet_oclr_opt )

    ## Test the L1 ceiling computation
    l1c <- with( params, l1c_oclr(X, l2, d, P, m) )
    m1 <- ftrain( params, l1=l1c )
    m2 <- ftrain( params, l1=l1c-0.0001 )
    expect_equal( sum(m1$w), 0 )
    expect_length( which(m2$w != 0), 1 )
})
