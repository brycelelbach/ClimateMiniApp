#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

#include "MayDay.H"

#include "LapackWrapper.H"
#include "Lapack.H"

#include "NamespaceHeader.H"


void LapackWrapper::applyBandMatrix(Real* const in, Real* const out, 
    LapackFactorization& A)
{
    // Sorry, need these for the fortran version of LAPACK
    char TRANS = 'N';
    int INCX = 1;
    int INCY = 1;
    int M = A.numCols(); // it's actually a square matrix
    int N = A.numCols(); // in compact band format
    int KL = A.numLower();
    int KU = 2*A.numUpper(); // FIXME - 2x seems to be necessary?
    int LDA = A.numRows();
    int INFO;

    // Call the banded matrix multiply routine
    Real alpha = 1;
    Real beta = 0;
    dgbmv_(&TRANS, &M, &N, &KL, &KU, &alpha, A.luPtr(), &LDA, in, 
        &INCX, &beta, out, &INCY);
}

void LapackWrapper::factorBandMatrix(LapackFactorization& A)
{
    int LDAB = A.numRows();
    int M = A.numCols(); // it's actually a square matrix
    int N = A.numCols(); // in compact band format

    int KL = A.numLower();
    int KU = A.numUpper();
    int INFO;

    // Factorization
    // dgtf2 is the unblocked version of dgbtrf
    dgbtf2_(&M, &N, &KL, &KU, A.luPtr(), &LDAB, A.pivotPtr(), &INFO);

    CH_assert(INFO == 0);
}


void LapackWrapper::solveBandMatrix(LapackFactorization& A, Real* const inout)
{
    // - check that the sizes of A, B are compatible
    int LDAB = A.numRows();
    int N = A.numCols(); // square matrix in compact band format

    int KL = A.numLower();
    int KU = A.numUpper();
    int INFO;

    // Solve using factorization
    char TRANS = 'N';
    int NRHS = 1;
    chombo_dgbtrs_(&TRANS, &N, &KL, &KU, &NRHS, A.luPtr(), 
                   &LDAB, A.pivotPtr(), inout, &N, &INFO);

    CH_assert(INFO == 0);
}

#include "NamespaceFooter.H"
