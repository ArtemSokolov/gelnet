% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/def.R
\name{rglz_nf}
\alias{rglz_nf}
\title{Alternative L1 regularizer}
\usage{
rglz_nf(nFeats, d = NULL)
}
\arguments{
\item{nFeats}{desired number of features with non-zero weights in the model}

\item{d}{p-by-1 vector of feature weights}
}
\value{
A regularizer definition that can be combined with a model definition using + operator
}
\description{
Defines an L1 regularizer that results in the desired number of non-zero feature weights
}
\details{
The corresponding regularization coefficient is determined through binary search
}
