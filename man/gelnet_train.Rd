% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gelnet.R
\name{gelnet_train}
\alias{gelnet_train}
\title{Trains a GELnet model}
\usage{
gelnet_train(modeldef, max_iter = 100L, eps = 1e-05, silent = FALSE,
  verbose = FALSE)
}
\arguments{
\item{modeldef}{model definition constructed through gelnet() arithmetic}

\item{max_iter}{maximum number of iterations}

\item{eps}{convergence precision}

\item{silent}{set to TRUE to suppress run-time output to stdout; overrides verbose (default: FALSE)}

\item{verbose}{set to TRUE to see extra output; is overridden by silent (default: FALSE)}
}
\value{
A GELNET model, expressed as a list with two elements:
\describe{
  \item{w}{p-by-1 vector of p model weights}
  \item{b}{scalar, bias term for the linear model (omitted for one-class models)}
}
}
\description{
Trains a model on the definition constructed by gelnet()
}
\details{
The training is performed through cyclical coordinate descent, and the optimization is terminated after the desired tolerance is achieved or after a maximum number of iterations.
}
