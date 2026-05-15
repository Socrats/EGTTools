/*
 * arpack_cwrapper.c — Thin C wrappers that call ARPACK Fortran routines.
 *
 * EasyBuild Autotools builds of arpack-ng compile the Fortran core but omit
 * the C-API wrapper layer (dnaupd_c, dneupd_c, etc.).  This file provides
 * those wrappers by calling the Fortran symbols directly.
 *
 * GFortran ABI (used by the foss toolchain):
 *   - Fortran routine names: lower-case with trailing underscore (dnaupd_)
 *   - CHARACTER arguments: pointer + hidden size_t length at the end,
 *     one per CHARACTER argument in declaration order.
 *
 * BSD-3-Clause (same licence as arpack-ng).
 */

#include <arpack/arpackdef.h>
#include <string.h>

/* Hidden character-length type for GFortran (size_t on 64-bit). */
typedef unsigned long fc_len_t;

/* ------------------------------------------------------------------ */
/* Fortran routine prototypes (GFortran name mangling)                 */
/* ------------------------------------------------------------------ */

/* Non-symmetric double */
extern void dnaupd_(a_int* ido, const char* bmat, a_int* n, const char* which,
                    a_int* nev, double* tol, double* resid, a_int* ncv,
                    double* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    double* workd, double* workl, a_int* lworkl, a_int* info,
                    fc_len_t bmat_len, fc_len_t which_len);
extern void dneupd_(a_int* rvec, const char* howmny, const a_int* select,
                    double* dr, double* di, double* z, a_int* ldz,
                    double* sigmar, double* sigmai, double* workev,
                    const char* bmat, a_int* n, const char* which,
                    a_int* nev, double* tol, double* resid, a_int* ncv,
                    double* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    double* workd, double* workl, a_int* lworkl, a_int* info,
                    fc_len_t howmny_len, fc_len_t bmat_len, fc_len_t which_len);

/* Non-symmetric float */
extern void snaupd_(a_int* ido, const char* bmat, a_int* n, const char* which,
                    a_int* nev, float* tol, float* resid, a_int* ncv,
                    float* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    float* workd, float* workl, a_int* lworkl, a_int* info,
                    fc_len_t bmat_len, fc_len_t which_len);
extern void sneupd_(a_int* rvec, const char* howmny, const a_int* select,
                    float* dr, float* di, float* z, a_int* ldz,
                    float* sigmar, float* sigmai, float* workev,
                    const char* bmat, a_int* n, const char* which,
                    a_int* nev, float* tol, float* resid, a_int* ncv,
                    float* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    float* workd, float* workl, a_int* lworkl, a_int* info,
                    fc_len_t howmny_len, fc_len_t bmat_len, fc_len_t which_len);

/* Symmetric double */
extern void dsaupd_(a_int* ido, const char* bmat, a_int* n, const char* which,
                    a_int* nev, double* tol, double* resid, a_int* ncv,
                    double* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    double* workd, double* workl, a_int* lworkl, a_int* info,
                    fc_len_t bmat_len, fc_len_t which_len);
extern void dseupd_(a_int* rvec, const char* howmny, const a_int* select,
                    double* d, double* z, a_int* ldz, double* sigma,
                    const char* bmat, a_int* n, const char* which,
                    a_int* nev, double* tol, double* resid, a_int* ncv,
                    double* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    double* workd, double* workl, a_int* lworkl, a_int* info,
                    fc_len_t howmny_len, fc_len_t bmat_len, fc_len_t which_len);

/* Symmetric float */
extern void ssaupd_(a_int* ido, const char* bmat, a_int* n, const char* which,
                    a_int* nev, float* tol, float* resid, a_int* ncv,
                    float* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    float* workd, float* workl, a_int* lworkl, a_int* info,
                    fc_len_t bmat_len, fc_len_t which_len);
extern void sseupd_(a_int* rvec, const char* howmny, const a_int* select,
                    float* d, float* z, a_int* ldz, float* sigma,
                    const char* bmat, a_int* n, const char* which,
                    a_int* nev, float* tol, float* resid, a_int* ncv,
                    float* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    float* workd, float* workl, a_int* lworkl, a_int* info,
                    fc_len_t howmny_len, fc_len_t bmat_len, fc_len_t which_len);

/* Complex float */
extern void cnaupd_(a_int* ido, const char* bmat, a_int* n, const char* which,
                    a_int* nev, float* tol, a_fcomplex* resid, a_int* ncv,
                    a_fcomplex* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    a_fcomplex* workd, a_fcomplex* workl, a_int* lworkl,
                    float* rwork, a_int* info,
                    fc_len_t bmat_len, fc_len_t which_len);
extern void cneupd_(a_int* rvec, const char* howmny, const a_int* select,
                    a_fcomplex* d, a_fcomplex* z, a_int* ldz, a_fcomplex* sigma,
                    a_fcomplex* workev, const char* bmat, a_int* n,
                    const char* which, a_int* nev, float* tol, a_fcomplex* resid,
                    a_int* ncv, a_fcomplex* v, a_int* ldv, a_int* iparam,
                    a_int* ipntr, a_fcomplex* workd, a_fcomplex* workl,
                    a_int* lworkl, float* rwork, a_int* info,
                    fc_len_t howmny_len, fc_len_t bmat_len, fc_len_t which_len);

/* Complex double */
extern void znaupd_(a_int* ido, const char* bmat, a_int* n, const char* which,
                    a_int* nev, double* tol, a_dcomplex* resid, a_int* ncv,
                    a_dcomplex* v, a_int* ldv, a_int* iparam, a_int* ipntr,
                    a_dcomplex* workd, a_dcomplex* workl, a_int* lworkl,
                    double* rwork, a_int* info,
                    fc_len_t bmat_len, fc_len_t which_len);
extern void zneupd_(a_int* rvec, const char* howmny, const a_int* select,
                    a_dcomplex* d, a_dcomplex* z, a_int* ldz, a_dcomplex* sigma,
                    a_dcomplex* workev, const char* bmat, a_int* n,
                    const char* which, a_int* nev, double* tol, a_dcomplex* resid,
                    a_int* ncv, a_dcomplex* v, a_int* ldv, a_int* iparam,
                    a_int* ipntr, a_dcomplex* workd, a_dcomplex* workl,
                    a_int* lworkl, double* rwork, a_int* info,
                    fc_len_t howmny_len, fc_len_t bmat_len, fc_len_t which_len);

/* ------------------------------------------------------------------ */
/* C-API implementations                                               */
/* ------------------------------------------------------------------ */

void dnaupd_c(a_int* ido, char const* bmat, a_int n, char const* which,
              a_int nev, double tol, double* resid, a_int ncv,
              double* v, a_int ldv, a_int* iparam, a_int* ipntr,
              double* workd, double* workl, a_int lworkl, a_int* info) {
    dnaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void dneupd_c(a_int rvec, char const* howmny, a_int const* select,
              double* dr, double* di, double* z, a_int ldz,
              double sigmar, double sigmai, double* workev,
              char const* bmat, a_int n, char const* which,
              a_int nev, double tol, double* resid, a_int ncv,
              double* v, a_int ldv, a_int* iparam, a_int* ipntr,
              double* workd, double* workl, a_int lworkl, a_int* info) {
    dneupd_(&rvec, howmny, select, dr, di, z, &ldz, &sigmar, &sigmai, workev,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(howmny), (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void snaupd_c(a_int* ido, char const* bmat, a_int n, char const* which,
              a_int nev, float tol, float* resid, a_int ncv,
              float* v, a_int ldv, a_int* iparam, a_int* ipntr,
              float* workd, float* workl, a_int lworkl, a_int* info) {
    snaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void sneupd_c(a_int rvec, char const* howmny, a_int const* select,
              float* dr, float* di, float* z, a_int ldz,
              float sigmar, float sigmai, float* workev,
              char const* bmat, a_int n, char const* which,
              a_int nev, float tol, float* resid, a_int ncv,
              float* v, a_int ldv, a_int* iparam, a_int* ipntr,
              float* workd, float* workl, a_int lworkl, a_int* info) {
    sneupd_(&rvec, howmny, select, dr, di, z, &ldz, &sigmar, &sigmai, workev,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(howmny), (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void dsaupd_c(a_int* ido, char const* bmat, a_int n, char const* which,
              a_int nev, double tol, double* resid, a_int ncv,
              double* v, a_int ldv, a_int* iparam, a_int* ipntr,
              double* workd, double* workl, a_int lworkl, a_int* info) {
    dsaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void dseupd_c(a_int rvec, char const* howmny, a_int const* select,
              double* d, double* z, a_int ldz, double sigma,
              char const* bmat, a_int n, char const* which,
              a_int nev, double tol, double* resid, a_int ncv,
              double* v, a_int ldv, a_int* iparam, a_int* ipntr,
              double* workd, double* workl, a_int lworkl, a_int* info) {
    dseupd_(&rvec, howmny, select, d, z, &ldz, &sigma,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(howmny), (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void ssaupd_c(a_int* ido, char const* bmat, a_int n, char const* which,
              a_int nev, float tol, float* resid, a_int ncv,
              float* v, a_int ldv, a_int* iparam, a_int* ipntr,
              float* workd, float* workl, a_int lworkl, a_int* info) {
    ssaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void sseupd_c(a_int rvec, char const* howmny, a_int const* select,
              float* d, float* z, a_int ldz, float sigma,
              char const* bmat, a_int n, char const* which,
              a_int nev, float tol, float* resid, a_int ncv,
              float* v, a_int ldv, a_int* iparam, a_int* ipntr,
              float* workd, float* workl, a_int lworkl, a_int* info) {
    sseupd_(&rvec, howmny, select, d, z, &ldz, &sigma,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, info,
            (fc_len_t)strlen(howmny), (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void cnaupd_c(a_int* ido, char const* bmat, a_int n, char const* which,
              a_int nev, float tol, a_fcomplex* resid, a_int ncv,
              a_fcomplex* v, a_int ldv, a_int* iparam, a_int* ipntr,
              a_fcomplex* workd, a_fcomplex* workl, a_int lworkl,
              float* rwork, a_int* info) {
    cnaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, rwork, info,
            (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void cneupd_c(a_int rvec, char const* howmny, a_int const* select,
              a_fcomplex* d, a_fcomplex* z, a_int ldz,
              a_fcomplex sigma, a_fcomplex* workev,
              char const* bmat, a_int n, char const* which,
              a_int nev, float tol, a_fcomplex* resid, a_int ncv,
              a_fcomplex* v, a_int ldv, a_int* iparam, a_int* ipntr,
              a_fcomplex* workd, a_fcomplex* workl, a_int lworkl,
              float* rwork, a_int* info) {
    cneupd_(&rvec, howmny, select, d, z, &ldz, &sigma, workev,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, rwork, info,
            (fc_len_t)strlen(howmny), (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void znaupd_c(a_int* ido, char const* bmat, a_int n, char const* which,
              a_int nev, double tol, a_dcomplex* resid, a_int ncv,
              a_dcomplex* v, a_int ldv, a_int* iparam, a_int* ipntr,
              a_dcomplex* workd, a_dcomplex* workl, a_int lworkl,
              double* rwork, a_int* info) {
    znaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, rwork, info,
            (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}

void zneupd_c(a_int rvec, char const* howmny, a_int const* select,
              a_dcomplex* d, a_dcomplex* z, a_int ldz,
              a_dcomplex sigma, a_dcomplex* workev,
              char const* bmat, a_int n, char const* which,
              a_int nev, double tol, a_dcomplex* resid, a_int ncv,
              a_dcomplex* v, a_int ldv, a_int* iparam, a_int* ipntr,
              a_dcomplex* workd, a_dcomplex* workl, a_int lworkl,
              double* rwork, a_int* info) {
    zneupd_(&rvec, howmny, select, d, z, &ldz, &sigma, workev,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, rwork, info,
            (fc_len_t)strlen(howmny), (fc_len_t)strlen(bmat), (fc_len_t)strlen(which));
}
