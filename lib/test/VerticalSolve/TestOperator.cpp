#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif


#include "TestOperator.H"

#include <assert.h>
#include "Lapack.H"
// #include <lapacke.h>

#include "NamespaceHeader.H"

TestOperator::TestOperator()
{;}

TestOperator::~TestOperator()
{;}

/// Creates a tridiagonal matrix for coef * D_xx
/// with homogeneous Neumann bc's
void 
TestOperator::
build2ndOrderOperator(band_matrix& a_A, Real& a_coef, Real& a_dx, int a_N)
{
  // Stencil for D_xx
  const int nbands = 5;
  const Real s[]={0.0, 1.0, -2.0, 1.0, 0.0};
  assert(nbands < a_A.maxbands);
  a_A.nbands = nbands;
  // const Real s[5]={0.0, 0.0, 1.0, 0.0, 0.0}; // for identity matrix

  Real coef = a_coef/(a_dx*a_dx);
  // Real coef = 1.0; // for identity matrix

  // 2nd Sub-diagonal of the matrix.
  a_A.d[0].resize(a_N-2, s[0]*coef);
  // 1st Sub-diagonal of the matrix.
  a_A.d[1].resize(a_N-1, s[1]*coef);
  // Diagonal of the matrix.
  a_A.d[2].resize(a_N, s[2]*coef);
  // 1st Super-diagonal of the matrix.
  a_A.d[3].resize(a_N-1, s[3]*coef);
  // 2nd Super-diagonal of the matrix.
  a_A.d[4].resize(a_N-2, s[4]*coef);

  // Fix the diagonal ends for the homogeneous Neumann bc's
  const Real bc = -1.0; 
  // const Real bc = 1.0; // for identity matrix
  a_A.d[2][0] = bc*coef;
  a_A.d[2][a_N-1] = bc*coef;
}


/// Creates a tridiagonal matrix for a backward Euler
/// heat equation step: (I - coef * D_xx) with homogeneous Neumann bc's
void 
TestOperator::
build2ndOrderSolver(band_matrix& a_A, Real& a_coef, Real& a_dx, int a_N)
{
  // Stencil for D_xx
  const int nbands = 5;
  const Real s[]={0.0, 1.0, -2.0, 1.0, 0.0};
  a_A.nbands = nbands;

  Real coef = a_coef/(a_dx*a_dx);
  // 2nd Sub-diagonal of the matrix.
  a_A.d[0].resize(a_N-2, -s[0]*coef);
  // 1st Sub-diagonal of the matrix.
  a_A.d[1].resize(a_N-1, -s[1]*coef);
  // Diagonal of the matrix.
  a_A.d[2].resize(a_N, 1.0 - s[2]*coef);
  // 1st Super-diagonal of the matrix.
  a_A.d[3].resize(a_N-1, -s[3]*coef);
  // 2nd Super-diagonal of the matrix.
  a_A.d[4].resize(a_N-2, -s[4]*coef);

  // Fix the diagonal ends for the homogeneous Neumann bc's
  const Real bc = -1.0;
  a_A.d[2][0] = 1.0 - bc*coef;
  a_A.d[2][a_N-1] = 1.0 - bc*coef;
}


/// Creates a tridiagonal matrix for coef * D_xx
/// with homogeneous Neumann bc's
void 
TestOperator::
build4thOrderOperator(band_matrix& a_A, Real& a_coef, Real& a_dx, int a_N)
{
  // Stencil for D_xx
  const Real s[5]={0.0, 1.0, -2.0, 1.0, 0.0};
  // const Real s[5]={0.0, 0.0, 1.0, 0.0, 0.0}; // for identity matrix

  Real coef = a_coef/(a_dx*a_dx);
  // Real coef = 1.0; // for identity matrix

  // 2nd Sub-diagonal of the matrix.
  a_A.d[0].resize(a_N-2, s[0]*coef);
  // 1st Sub-diagonal of the matrix.
  a_A.d[1].resize(a_N-1, s[1]*coef);
  // Diagonal of the matrix.
  a_A.d[2].resize(a_N, s[2]*coef);
  // 1st Super-diagonal of the matrix.
  a_A.d[3].resize(a_N-1, s[3]*coef);
  // 2nd Super-diagonal of the matrix.
  a_A.d[4].resize(a_N-2, s[4]*coef);

  // Fix the diagonal ends for the homogeneous Neumann bc's
  const Real bc = -1.0; 
  // const Real bc = 1.0; // for identity matrix
  a_A.d[2][0] = bc*coef;
  a_A.d[2][a_N-1] = bc*coef;
}


/// Creates a 7-diagonal matrix for a backward Euler
/// heat equation step: (I - coef * D_xx) with homogeneous Neumann bc's
void 
TestOperator::
build4thOrderSolver(band_matrix& a_A, Real& a_coef, Real& a_dx, int a_N)
{
  // Stencil for D_xx
  const int nbands = 7;
  // const Real s[]={0.0, -1.0, 16.0, -30.0, 16.0, -1.0, 0.0};
  const Real s[]={0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0};
  a_A.nbands = nbands;

  // FIXME Real coef = a_coef/(12.0*a_dx*a_dx);
  Real coef = a_coef/(1.0*a_dx*a_dx);
  // 3rd Sub-diagonal of the matrix.
  a_A.d[0].resize(a_N-3, -s[0]*coef);
  // 2nd Sub-diagonal of the matrix.
  a_A.d[1].resize(a_N-2, -s[1]*coef);
  // 1st Sub-diagonal of the matrix.
  a_A.d[2].resize(a_N-1, -s[2]*coef);
  // Diagonal of the matrix.
  a_A.d[3].resize(a_N, 1.0 - s[3]*coef);
  // 1st Super-diagonal of the matrix.
  a_A.d[4].resize(a_N-1, -s[4]*coef);
  // 2nd Super-diagonal of the matrix.
  a_A.d[5].resize(a_N-2, -s[5]*coef);
  // 3rd Super-diagonal of the matrix.
  a_A.d[6].resize(a_N-3, -s[6]*coef);

  // Fix the diagonal ends for the homogeneous Neumann bc's
  const Real bc = -1.0;
  a_A.d[3][0] = 1.0 - bc*coef;
  a_A.d[3][a_N-1] = 1.0 - bc*coef;
}


/// Applies the banded matrix A as an operator
/// to each x row of the FAB, 
/// rhs := alpha * A * state + beta * rhs
void 
TestOperator::
applyBandedMatrix(FArrayBox& a_state,
                  FArrayBox& a_rhs,
                  const band_matrix& a_A,
                  Real a_alpha, 
                  Real a_beta) 
{
  CH_TIME("TestOperator::applyBandedMatrix");

  IntVect size = a_state.box().size();
  int size0 = size[0]; // x direction stencil
  assert(size0 == a_rhs.box().size(0));

  // Create local storage for banded matrix format
  int KL = 2; // for 5-band matrix
  int KU = 2;
  int nA = 1 + KL + KU;
  Real* A = new Real[nA * size0]; 
  for (int i=0; i < nA * size0; i++)
    A[i] = -CH_BADVAL;

  // Copy over the values from a_A in weird band format
  int diagIx = 2; // index for the diagonal array
  for (int d=0; d < a_A.nbands; d++)
  {
    int invd = a_A.nbands - d - 1;
    int size = size0 - abs(invd-diagIx);
    for (int i=0; i < size; i++)
    {
      int offset = (max(invd-diagIx,0) + i) * nA;
      // (a_A.nbands-1-d)*size0 + max(d-diagIx,0);
      int ix = d + offset; // where the first entry goes
      A[ix] = a_A.d[invd][i];
    }
  }

  /*
  for (int i=0; i < nA * size0; i++)
  {
    // if (i % size0 == 0)
    //   printf("\n  A[%d,.] (%d) = [", i / size0, i);
    printf("%6.2e, ", A[i]);
    if (i % nA == nA-1)
      printf("\n");
  }
  */

  IntVect lower = a_rhs.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* state = &a_state(IntVect(lower[0], j, k)); 
      Real* rhs = &a_rhs(IntVect(lower[0], j, k)); 

      // Sorry, need these for the fortran version of LAPACK
      char TRANS = 'N';
      int N = size0;
      int LDA = nA;
      int INCX = 1;
      int INCY = 1;

      // Call the banded matrix multiply routine
      dgbmv_(&TRANS, &size0, &N, &KL, &KU, 
          &a_alpha, A, &LDA, state, &INCX, &a_beta, rhs, &INCY);
    }

  delete A;
}


/// Does a tridiagonal solve for a backward Euler
/// heat equation step: (I - coef * D_xx) state = rhs
void 
TestOperator::
implicitSolve(FArrayBox& a_state,
              FArrayBox& a_rhs,
              const band_matrix& a_A)
{
  CH_TIME("TestOperator::implicitSolve");

  IntVect size = a_state.box().size();
  int size0 = size[0]; // x direction stencil
  assert(size0 == a_rhs.box().size(0));

  int* ipiv = new int[size0];
  Real* Atmp = new Real[size0];

  IntVect lower = a_rhs.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* rhs = &a_rhs(IntVect(lower[0], j, k)); 
      // LAPACK overwrites things, so need to reset
      band_matrix Atmp; 
      for (int diag=0; diag < a_A.nbands; diag++)
        Atmp.d[diag] = a_A.d[diag];

      // Sorry, need these for the fortran version of LAPACK
      int info = 0;
      int NRHS = 1;
      int LDB = size0;
      dgtsv_(
          // LAPACK_ROW_MAJOR, // matrix format
          &size0, // matrix order
          &NRHS, // # of right hand sides 
          Atmp.d[2].data(), // 1st subdiagonal part
          Atmp.d[3].data(), // diagonal part
          Atmp.d[4].data(), // 1st superdiagonal part
          rhs, // column to solve 
          &LDB, // leading dimension of RHS
          &info
          );
      /*
SUBROUTINE DGBSV( N, KL, KU, NRHS, AB, LDAB, IPIV, B, LDB, INFO )

      int KU = 1;
      int KL = 1;
      dgtsv_(
          &size0, // matrix order
          &KU, // # of upper diagonals 
          &KL, // # of lower diagonals 
          &NRHS, // # of right hand sides 

          Atmp.d[2].data(), // 1st subdiagonal part
          Atmp.d[3].data(), // diagonal part
          Atmp.d[4].data(), // 1st superdiagonal part

          rhs, // column to solve 
          &LDB, // leading dimension of RHS
          &info
          );
      */

      if (info != 0)
        pout() << "Lapack info=" << info << endl;
      assert(info == 0);

      Real* state = &a_state(IntVect(lower[0], j, k)); 
      for (int i=0; i < size0; i++)
        state[i] = rhs[i];
    }
}


void 
TestOperator::
setExact(FArrayBox& a_exact,
         int a_kx,
         Real a_dx, // x direction spacing
         Real a_coef)
{
  CH_TIME("TestOperator::setExact");

  const IntVect size = a_exact.box().size();
  const int size0 = size[0]; // x direction size
  const Real kxpi = M_PI*(Real) a_kx;

  const IntVect lower = a_exact.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* exact = &a_exact(IntVect(lower[0], j, k)); 
      for (int i=0; i < size0; i++)
      {
        // Calculate the average of cos(kx pi x)
        Real xhi = a_dx*(Real)(i+1);
        Real xlo = a_dx*(Real)i;
        exact[i] = a_coef*(sin(kxpi*xhi) - sin(kxpi*xlo))/(kxpi*a_dx);
      }
    }
}
