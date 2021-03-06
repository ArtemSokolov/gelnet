% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/def.R
\name{model_oclr}
\alias{model_oclr}
\title{One-class logistic regression}
\usage{
model_oclr(nonneg = FALSE)
}
\arguments{
\item{nonneg}{set to TRUE to enforce non-negativity constraints on the weights (default: FALSE)}
}
\value{
A GELnet task definition that can be combined with gelnet() output
}
\description{
Defines a one-class logistic regression (OCLR) task
}
\details{
The OCLR objective function is defined as
\deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
 where
\deqn{ s_i = w^T x_i }
}
