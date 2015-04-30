#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

#ifndef _LAPACK_H_
#define _LAPACK_H_

#include "NamespaceHeader.H"

#define LAPACK_ROW_MAJOR 101

// Forward declaration of lapack routines
extern "C" 
{
    // Tridiagonal solve
    void dgtsv_(int* N, int* NRHS, double* DL, double* D,
                double* DU, double* B, int* LDB, int* INFO);

    // Triangular solve
    void dtrtrs_(char* UPLO, char* TRANS, char* DIAG, int* N, int* NRHS, 
                 double* A, int* LDA, double* B, int* LDB, int* INFO);

    // Band matrix LU factorization
    void dgbtrf_(int* M, int* N, int* KL, int* KU, double* AB, 
                 int* LDAB, int* IPIV, int* INFO);

    // Band matrix LU solve
    void dgbtrs_(char* TRANS, int* N, int* KL, int* KU, int* NRHS, double* AB, 
                 int* LDAB, int* IPIV, double* B, int* LDB, int* INFO);
}

#include "NamespaceFooter.H"
#endif
