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

/// A struct to hold the diagonals of a banded matrix
struct band_matrix
{
  const static int nbands = 5;
  std::vector<Real> d[nbands];
};


void 
TestOperator::
implicitSolve4thOrder(FArrayBox& a_state,
                      FArrayBox& a_rhs,
                      Real a_coef)
{
  CH_TIME("TestOperator::implicitSolve4thOrder");

  IntVect size = a_state.box().size();
  int size0 = size[0]; // x direction stencil
  assert(size0 == a_rhs.box().size(0));

  band_matrix A;
  // Stencil for -d_xx
  const Real s[5]={0.0, -1.0, 2.0, -1.0, 0.0};

  Real dx = 1.0/(Real) size0;
  Real coef = a_coef/(dx*dx);
  // 2nd Sub-diagonal of the matrix.
  A.d[0].resize(size0-2, s[0]*coef);
  // 1st Sub-diagonal of the matrix.
  A.d[1].resize(size0-1, s[1]*coef);
  // Diagonal of the matrix.
  A.d[2].resize(size0, 1.0 + s[2]*coef);
  // 1st Super-diagonal of the matrix.
  A.d[3].resize(size0-1, s[3]*coef);
  // 2nd Super-diagonal of the matrix.
  A.d[4].resize(size0-2, s[4]*coef);

  // Fix the diagonal ends for the bc's
  const Real bc = 1.0;
  A.d[2][0] = 1.0 + bc*coef;
  A.d[2][size0-1] = 1.0 + bc*coef;

/*
  pout() << "Diagonals of A:" << endl;
  for (int i=0; i < size0; i++)
  {
    if (i>0) 
      pout() << "lower[" << i << "] = " << A.d[3][i-1] << endl;
    pout() << "diag[" << i << "] = " << A.d[2][i] << endl;
    if (i<(size0-1))
      pout() << "upper[" << i << "] = " << A.d[1][i] << endl;
  }
*/

  IntVect lower = a_rhs.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* rhs = &a_rhs(IntVect(lower[0], j, k)); 
      // LAPACK overwrites things, so need to reset
      band_matrix Atmp; 
      for (int diag=0; diag < A.nbands; diag++)
        Atmp.d[diag] = A.d[diag];

      // Sorry, need these for the fortran version of LAPACK
      int info = 0;
      int NRHS = 1;
      int LDB = size0;
      dgtsv_(
          // LAPACK_ROW_MAJOR, // matrix format
          &size0, // matrix order
          &NRHS, // # of right hand sides 
          Atmp.d[1].data(), // 1st subdiagonal part
          Atmp.d[2].data(), // diagonal part
          Atmp.d[3].data(), // 1st superdiagonal part
          rhs, // column to solve 
          &LDB, // leading dimension of RHS
          &info
          );

      if (info != 0)
        pout() << "Lapack info=" << info << endl;
      assert(info == 0);

      Real* state = &a_state(IntVect(lower[0], j, k)); 
      for (int i=0; i < size0; i++)
        state[i] = rhs[i];
    }
}


/// Does a tridiagonal solve
void 
TestOperator::
implicitSolve2ndOrder(FArrayBox& a_state,
                      FArrayBox& a_rhs,
                      Real a_coef)
{
  CH_TIME("TestOperator::implicitSolve2ndOrder");

  IntVect size = a_state.box().size();
  int size0 = size[0]; // x direction stencil
  assert(size0 == a_rhs.box().size(0));

  band_matrix A;
  // Stencil for -d_xx
  const Real s[5]={0.0, -1.0, 2.0, -1.0, 0.0};

  Real dx = 1.0/(Real) size0;
  Real coef = a_coef/(dx*dx);
  // 2nd Sub-diagonal of the matrix.
  A.d[0].resize(size0-2, s[0]*coef);
  // 1st Sub-diagonal of the matrix.
  A.d[1].resize(size0-1, s[1]*coef);
  // Diagonal of the matrix.
  A.d[2].resize(size0, 1.0 + s[2]*coef);
  // 1st Super-diagonal of the matrix.
  A.d[3].resize(size0-1, s[3]*coef);
  // 2nd Super-diagonal of the matrix.
  A.d[4].resize(size0-2, s[4]*coef);

  // Fix the diagonal ends for the bc's
  const Real bc = 1.0;
  A.d[2][0] = 1.0 + bc*coef;
  A.d[2][size0-1] = 1.0 + bc*coef;

/*
  pout() << "Diagonals of A:" << endl;
  for (int i=0; i < size0; i++)
  {
    if (i>0) 
      pout() << "lower[" << i << "] = " << A.d[3][i-1] << endl;
    pout() << "diag[" << i << "] = " << A.d[2][i] << endl;
    if (i<(size0-1))
      pout() << "upper[" << i << "] = " << A.d[1][i] << endl;
  }
*/

  IntVect lower = a_rhs.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* rhs = &a_rhs(IntVect(lower[0], j, k)); 
      // LAPACK overwrites things, so need to reset
      band_matrix Atmp; 
      for (int diag=0; diag < A.nbands; diag++)
        Atmp.d[diag] = A.d[diag];

      // Sorry, need these for the fortran version of LAPACK
      int info = 0;
      int NRHS = 1;
      int LDB = size0;
      dgtsv_(
          // LAPACK_ROW_MAJOR, // matrix format
          &size0, // matrix order
          &NRHS, // # of right hand sides 
          Atmp.d[1].data(), // 1st subdiagonal part
          Atmp.d[2].data(), // diagonal part
          Atmp.d[3].data(), // 1st superdiagonal part
          rhs, // column to solve 
          &LDB, // leading dimension of RHS
          &info
          );

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
         Real a_coef)
{
  CH_TIME("TestOperator::setExact");

  const IntVect size = a_exact.box().size();
  const int size0 = size[0]; // x direction size
  const Real dx = 1.0 / (Real) size0; // x direction spacing
  const Real kxpi = M_PI*(Real) a_kx;

  const IntVect lower = a_exact.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* exact = &a_exact(IntVect(lower[0], j, k)); 
      for (int i=0; i < size0; i++)
      {
        // Calculate the average of cos(kx pi x)
        Real xhi = dx*(Real)(i+1);
        Real xlo = dx*(Real)i;
        exact[i] = a_coef*(sin(kxpi*xhi) - sin(kxpi*xlo))/(kxpi*dx);
      }
    }
}
