% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{gelnet_lin_opt}
\alias{gelnet_lin_opt}
\title{GELnet optimizer for linear regression}
\usage{
gelnet_lin_opt(X, z, l1, l2, max_iter = 100L, eps = 1e-05,
  fix_bias = FALSE, silent = FALSE, verbose = FALSE,
  nonneg = FALSE, w_init = NULL, b_init = NULL, a = NULL,
  d = NULL, P = NULL, m = NULL)
}
\arguments{
\item{X}{n-by-p matrix of n samples in p dimensions}

\item{z}{n-by-1 vector of response values}

\item{l1}{coefficient for the L1-norm penalty}

\item{l2}{coefficient for the L2-norm penalty}

\item{max_iter}{maximum number of iterations}

\item{eps}{convergence precision}

\item{fix_bias}{set to TRUE to prevent the bias term from being updated (default: FALSE)}

\item{silent}{set to TRUE to suppress run-time output; overwrites verbose (default: FALSE)}

\item{verbose}{set to TRUE to see extra output; is overwritten by silent (default: FALSE)}

\item{nonneg}{set to TRUE to enforce non-negativity constraints on the weights (default: FALSE )}

\item{w_init}{initial parameter estimate for the weights}

\item{b_init}{initial parameter estimate for the bias term}

\item{a}{n-by-1 vector of sample weights}

\item{d}{p-by-1 vector of feature weights}

\item{P}{p-by-p feature association penalty matrix}

\item{m}{p-by-1 vector of translation coefficients}
}
\value{
A list with two elements:
\describe{
  \item{w}{p-by-1 vector of p model weights}
  \item{b}{scalar, bias term for the linear model}
}
}
\description{
Constructs a GELnet model for linear regression using coordinate descent.
}
\details{
The method operates through cyclical coordinate descent.
The optimization is terminated after the desired tolerance is achieved, or after a maximum number of iterations.
}
