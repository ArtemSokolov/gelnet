## Utility functions
##
## by Artem Sokolov

#' Generate a graph Laplacian
#'
#' Generates a graph Laplacian from the graph adjacency matrix.
#'
#' A graph Laplacian is defined as:
#' \eqn{ l_{i,j} = deg( v_i ) }, if \eqn{ i = j };
#' \eqn{ l_{i,j} = -1 }, if \eqn{ i \neq j } and \eqn{v_i} is adjacent to \eqn{v_j};
#' and \eqn{ l_{i,j} = 0 }, otherwise
#'
#' @param A n-by-n adjacency matrix for a graph with n nodes
#' @return The n-by-n Laplacian matrix of the graph
#' @seealso \code{\link{adj2nlapl}}
#' @export
adj2lapl <- function( A )
  {
    n <- nrow(A)
    stopifnot( ncol(A) == n )

    ## Compute the off-diagonal entries
    L <- -A
    diag(L) <- 0

    ## Compute the diagonal entries
    ## Degree of a node: sum of weights on the edges
    s <- apply( L, 2, sum )
    diag(L) <- -s	## Negative because L == -A
    L
  }

#' Generate a normalized graph Laplacian
#'
#' Generates a normalized graph Laplacian from the graph adjacency matrix.
#'
#' A normalized graph Laplacian is defined as:
#' \eqn{ l_{i,j} = 1 }, if \eqn{ i = j };
#' \eqn{ l_{i,j} = - 1 / \sqrt{ deg(v_i) deg(v_j) } }, if \eqn{ i \neq j } and \eqn{v_i} is adjacent to \eqn{v_j};
#' and \eqn{ l_{i,j} = 0 }, otherwise
#'
#' @param A n-by-n adjacency matrix for a graph with n nodes
#' @return The n-by-n Laplacian matrix of the graph
#' @seealso \code{\link{adj2nlapl}}
#' @export
adj2nlapl <- function(A)
  {
    n <- nrow(A)
    stopifnot( ncol(A) == n )

    ## Zero out the diagonal
    diag(A) <- 0

    ## Degree of a node: sum of weights on the edges
    d <- 1 / apply( A, 2, sum )
    d <- sqrt(d)

    ## Account for "free-floating" nodes
    j <- which( is.infinite(d) )
    d[j] <- 0

    ## Compute the non-normalized Laplacian
    L <- adj2lapl( A )

    ## Normalize
    res <- t( L*d ) * d
    rownames(res) <- rownames(A)
    colnames(res) <- rownames(res)
    res
  }

#' Perturbs a GELnet model
#'
#' Given a linear model, perturbs its i^th coefficient by delta
#'
#' @param model The model to perturb
#' @param i Index of the coefficient to modify, or 0 for the bias term
#' @param delta The value to perturb by
#' @return Modified GELnet model
#' @export
perturb.gelnet <- function( model, i, delta )
{
    res <- model
    if( i == 0 ) res$b <- res$b + delta
    else res$w[i] <- res$w[i] + delta
    res
}

