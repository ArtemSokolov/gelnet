// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// l1c_lin
double l1c_lin(arma::mat X, arma::vec z, double l2, Nullable<NumericVector> a, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_l1c_lin(SEXP XSEXP, SEXP zSEXP, SEXP l2SEXP, SEXP aSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type z(zSEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type a(aSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(l1c_lin(X, z, l2, a, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// l1c_blr
double l1c_blr(arma::mat X, arma::Col<int> y, double l2, bool balanced, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_l1c_blr(SEXP XSEXP, SEXP ySEXP, SEXP l2SEXP, SEXP balancedSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::Col<int> >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< bool >::type balanced(balancedSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(l1c_blr(X, y, l2, balanced, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// l1c_oclr
double l1c_oclr(arma::mat X, double l2, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_l1c_oclr(SEXP XSEXP, SEXP l2SEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(l1c_oclr(X, l2, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// gelnet_oclr_obj
double gelnet_oclr_obj(arma::vec w, arma::mat X, double l1, double l2, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_gelnet_oclr_obj(SEXP wSEXP, SEXP XSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(gelnet_oclr_obj(w, X, l1, l2, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// gelnet_lin_obj
double gelnet_lin_obj(arma::vec w, double b, arma::mat X, arma::vec z, double l1, double l2, Nullable<NumericVector> a, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_gelnet_lin_obj(SEXP wSEXP, SEXP bSEXP, SEXP XSEXP, SEXP zSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP aSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type z(zSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type a(aSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(gelnet_lin_obj(w, b, X, z, l1, l2, a, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// gelnet_blr_obj
double gelnet_blr_obj(arma::vec w, double b, arma::mat X, arma::Col<int> y, double l1, double l2, bool balanced, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_gelnet_blr_obj(SEXP wSEXP, SEXP bSEXP, SEXP XSEXP, SEXP ySEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP balancedSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::Col<int> >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< bool >::type balanced(balancedSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(gelnet_blr_obj(w, b, X, y, l1, l2, balanced, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// gelnet_lin_opt
List gelnet_lin_opt(arma::mat X, arma::vec z, double l1, double l2, int max_iter, double eps, bool fix_bias, bool silent, bool verbose, bool nonneg, Nullable<NumericVector> w_init, Nullable<double> b_init, Nullable<NumericVector> a, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_gelnet_lin_opt(SEXP XSEXP, SEXP zSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP max_iterSEXP, SEXP epsSEXP, SEXP fix_biasSEXP, SEXP silentSEXP, SEXP verboseSEXP, SEXP nonnegSEXP, SEXP w_initSEXP, SEXP b_initSEXP, SEXP aSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type z(zSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< bool >::type fix_bias(fix_biasSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type w_init(w_initSEXP);
    Rcpp::traits::input_parameter< Nullable<double> >::type b_init(b_initSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type a(aSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(gelnet_lin_opt(X, z, l1, l2, max_iter, eps, fix_bias, silent, verbose, nonneg, w_init, b_init, a, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// gelnet_blr_opt
List gelnet_blr_opt(arma::mat X, arma::Col<int> y, double l1, double l2, int max_iter, double eps, bool silent, bool verbose, bool balanced, bool nonneg, Nullable<NumericVector> w_init, Nullable<double> b_init, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_gelnet_blr_opt(SEXP XSEXP, SEXP ySEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP max_iterSEXP, SEXP epsSEXP, SEXP silentSEXP, SEXP verboseSEXP, SEXP balancedSEXP, SEXP nonnegSEXP, SEXP w_initSEXP, SEXP b_initSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::Col<int> >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type balanced(balancedSEXP);
    Rcpp::traits::input_parameter< bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type w_init(w_initSEXP);
    Rcpp::traits::input_parameter< Nullable<double> >::type b_init(b_initSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(gelnet_blr_opt(X, y, l1, l2, max_iter, eps, silent, verbose, balanced, nonneg, w_init, b_init, d, P, m));
    return rcpp_result_gen;
END_RCPP
}
// gelnet_oclr_opt
List gelnet_oclr_opt(arma::mat X, double l1, double l2, int max_iter, double eps, bool silent, bool verbose, bool nonneg, Nullable<NumericVector> w_init, Nullable<NumericVector> d, Nullable<NumericMatrix> P, Nullable<NumericVector> m);
RcppExport SEXP _gelnet_gelnet_oclr_opt(SEXP XSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP max_iterSEXP, SEXP epsSEXP, SEXP silentSEXP, SEXP verboseSEXP, SEXP nonnegSEXP, SEXP w_initSEXP, SEXP dSEXP, SEXP PSEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< bool >::type silent(silentSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type w_init(w_initSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d(dSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type P(PSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(gelnet_oclr_opt(X, l1, l2, max_iter, eps, silent, verbose, nonneg, w_init, d, P, m));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gelnet_l1c_lin", (DL_FUNC) &_gelnet_l1c_lin, 7},
    {"_gelnet_l1c_blr", (DL_FUNC) &_gelnet_l1c_blr, 7},
    {"_gelnet_l1c_oclr", (DL_FUNC) &_gelnet_l1c_oclr, 5},
    {"_gelnet_gelnet_oclr_obj", (DL_FUNC) &_gelnet_gelnet_oclr_obj, 7},
    {"_gelnet_gelnet_lin_obj", (DL_FUNC) &_gelnet_gelnet_lin_obj, 10},
    {"_gelnet_gelnet_blr_obj", (DL_FUNC) &_gelnet_gelnet_blr_obj, 10},
    {"_gelnet_gelnet_lin_opt", (DL_FUNC) &_gelnet_gelnet_lin_opt, 16},
    {"_gelnet_gelnet_blr_opt", (DL_FUNC) &_gelnet_gelnet_blr_opt, 15},
    {"_gelnet_gelnet_oclr_opt", (DL_FUNC) &_gelnet_gelnet_oclr_opt, 12},
    {NULL, NULL, 0}
};

RcppExport void R_init_gelnet(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
