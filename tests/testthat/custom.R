## Custom expectation functions and generators

## A semi-lifted purrr::partial
## prms - list of preset parameters
## ...  - dynamic parameters
partial2 <- function( f, prms, ... )
{
    p <- purrr::list_modify( prms, ... )
    do.call( purrr::partial, c(list(f),p) )
}

## Ensures that training found the optimum by looking in the immediate
##  neighborhood of the solution model
## model - a model object, as returned by, e.g., gelnet_lin_opt()
## fobj - the matching objective function to minimize
## hasBias - set to TRUE if the model has a bias term that needs to be tested
expect_optimal <- function( model, fobj, hasBias = TRUE )
{
    ## Predefine the error message
    msg <- "Model is not optimized along component "

    ## Capture the actual value and its label in the parent env
    act <- quasi_label( rlang::enquo(model) )

    ## Compute the objective function value of the trained model
    act$p <- length(act$val$w)
    act$obj <- fobj( act$val )

    ## Consider the immediate neighborhood of the solution
    iStart <- as.integer( !hasBias )
    for( i in iStart:act$p )
    {
        act$objn <- fobj( perturb.gelnet(act$val, i, -0.001) )
        act$objp <- fobj( perturb.gelnet(act$val, i, 0.001) )
        expect( act$objp > act$obj, stringr::str_c(msg, i) )
        expect( act$objn > act$obj, stringr::str_c(msg, i) )
    }

    ## Return the actual value (invisibly for pipe usage)
    invisible(act$val)
}

## A generalization of expect_optimal() that compares model optimality
##  in the context of other models / objective functions
## lmd - list of models
## lfn - list of matching function objectives
expect_relopt <- function( lmd, lfn )
{
    ## Capture the actual value and its label in the parent env
    act <- quasi_label( rlang::enquo(lmd) )

    ## Verify the arguments
    expect( length(lmd) == length(lfn),
           stringr::str_c("Each model in ", act$lab, " must have a matching objective function") )

    ## Traverse the models. For each, ensure it is more optimal for the
    ## corresponding objective function than other models in the list
    for( i in 1:length(lmd) )
    {
        ## Apply the i^th objective function to each model
        v <- purrr::map( lmd, lfn[[i]] )

        ## Ensure that the i^th model produces the lowest object value
        expect( which.min( v ) == i,
               glue::glue( "Model {i} in {act$lab} is outperformed on its ",
                          "objective function value" ) )
    }

    ## Return the actual value (invisibly for pipe usage)
    invisible(act$val)
}

