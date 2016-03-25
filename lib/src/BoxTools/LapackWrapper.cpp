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

/* Form y := A*x */
int chombo_dgbmv(
    int m
  , int n
  , int kl
  , int ku
  , double* a
  , int lda
  , double* x
  , double* y
    )
{
  if (m == 0 || n == 0)
    return 0;

  for (int i = 0; i < m; ++i) 
    y[i] = 0.;

  for (int j = 0; j < n; ++j) {
    if (x[j] != 0.) {
      #pragma simd
      for (int i = std::max(0, j - ku); i < std::min(m, j + 1 + kl); ++i) {
        y[i] += x[j] * a[(ku - j + i) + j * lda];
      }
    }
  }

  return 0;
}

void LapackWrapper::applyBandMatrix(Real* in, Real* out, 
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
    int INFO = 0;

    // Call the banded matrix multiply routine
    Real alpha = 1;
    Real beta = 0;

//    dgbmv_(&TRANS, &M, &N, &KL, &KU, &alpha, A.luPtr(), &LDA, in, 
//        &INCX, &beta, out, &INCY);

    chombo_dgbmv(M, N, KL, KU, A.luPtr(), LDA, in, out); 
}

void LapackWrapper::factorBandMatrix(LapackFactorization& A)
{
    int LDAB = A.numRows();
    int M = A.numCols(); // it's actually a square matrix
    int N = A.numCols(); // in compact band format

    int KL = A.numLower();
    int KU = A.numUpper();
    int INFO = 0;

    // Factorization
    // dgtf2 is the unblocked version of dgbtrf
    dgbtf2_(&M, &N, &KL, &KU, A.luPtr(), &LDAB, A.pivotPtr(), &INFO);

    CH_assert(INFO == 0);
}


void LapackWrapper::solveBandMatrix(LapackFactorization& A, Real* inout)
{
    int LDAB = A.numRows();
    int N = A.numCols(); // square matrix in compact band format

    int KL = A.numLower();
    int KU = A.numUpper();
    int INFO = 0;

    // Solve using factorization
    char TRANS = 'N';
    int NRHS = 1;
    chombo_dgbtrs_(&TRANS, &N, &KL, &KU, &NRHS, A.luPtr(), 
                   &LDAB, A.pivotPtr(), inout, &N, &INFO);

    CH_assert(INFO == 0);
}

#include "NamespaceFooter.H"
