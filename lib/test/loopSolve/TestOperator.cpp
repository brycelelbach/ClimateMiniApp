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
#include "LapackFactorization.H"
#include "LapackWrapper.H"
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
/// heat equation step: (c1 * I - c2 * D_xx) with homogeneous Neumann bc's
void 
TestOperator::
build4thOrderBanded(LapackFactorization& a_A,
    Real a_c1, Real a_c2, Real& a_dx, int a_N)
{
  // Real coef = a_c2/(12.0*a_dx*a_dx); // for 4th-order
  // const Real s[]={0.0, -1.0, 16.0, -30.0, 16.0, -1.0, 0.0};
  Real coef = a_c2/(1.0*a_dx*a_dx); // for 2nd-order
  const Real s[]={0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0};
  // Define it
  int KU = 3;
  int KL = 3;
  a_A.define(a_N, KL, KU);
  a_A.setZero();

  int ncol = a_A.numCols();
  for (int col=0; col < ncol; col++) 
    for (int ix=-KU; ix <= KL; ix++)
    {
      int row = col + ix;
      if ((row >= 0) && (row < ncol))
        a_A(row, col) = ((row == col) ? a_c1 : 0) + s[ix+KU]*coef;
        // a_A(row, col) = s[ix+KU]*coef;
    }

  // Fix the diagonal ends for the homogeneous Neumann bc's
  // For 2nd-order operator
  const Real bc = -1.0;
  a_A(0,0) = a_c1 + bc*coef;
  a_A(ncol-1,ncol-1) = a_c1 + bc*coef;

  // Fix the diagonal ends for the homogeneous Neumann bc's
  // For 4th-order operator
  // The gradient flux for the first interior face
  /*
  const int lenG = 4;
  const Real sG1[] = {-145.0, 159.0, -15.0, 1.0};
  const Real coefG1 = a_c2/(120*a_dx*a_dx);
  // The gradient flux for the second interior (regular) face
  const Real sG2[] = {1.0, -15.0, 15.0, -1.0};
  const Real coefG2 = a_c2/(12*a_dx*a_dx);
  for (int ix=0; ix < lenG; ix++)
  {
    a_A(0,ix) = ((ix == 0) ? a_c1 : 0) + sG1[ix] * coefG1;
    a_A(1,ix) = ((ix == 1) ? a_c1 : 0) + sG2[ix] * coefG2 - sG1[ix] * coefG1;
    int ixflip = ncol-ix-1;
    a_A(ncol-1,ixflip) = ((ix == 0) ? a_c1 : 0) + sG1[ix] * coefG1;
    a_A(ncol-2,ixflip) = 
      ((ix == 1) ? a_c1 : 0) + sG2[ix] * coefG2 - sG1[ix] * coefG1;
  }
  */

  // a_A.printBandedMatrix();
}


/// Applies the banded matrix A as an operator
/// to each x row of the FAB 
/// rhs = A * state
void 
TestOperator::
applyBandedMatrix(FArrayBox& a_state, FArrayBox& a_rhs,
    LapackFactorization& a_A)
{
  CH_TIME("TestOperator::applyBandedMatrix");

  IntVect size = a_state.box().size();
  int size0 = size[0]; // x direction stencil
  assert(size0 == a_rhs.box().size(0));

  IntVect lower = a_rhs.box().smallEnd();
  for (int j = lower[1]; j < size[1]; j++)
    for (int k = lower[2]; k < size[2]; k++)
    {
      Real* state = &a_state(IntVect(lower[0], j, k)); 
      Real* rhs = &a_rhs(IntVect(lower[0], j, k)); 
      LapackWrapper::applyBandMatrix(state, rhs, a_A);
    }
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

      if (info != 0)
        pout() << "Lapack info=" << info << endl;
      assert(info == 0);

      Real* state = &a_state(IntVect(lower[0], j, k)); 
      for (int i=0; i < size0; i++)
        state[i] = rhs[i];
    }
}


/// Does a banded solve for a backward Euler
/// heat equation step: (I - coef * D_zz) state = rhs
void 
TestOperator::
implicitSolveBanded(
    FArrayBox& a_state,
    FArrayBox& a_rhs,
    Box a_box,
    LapackFactorization& a_AB)
{
  CH_TIME("TestOperator::implicitSolveBanded");

  int N = a_box.size(2); // z-direction stencil
  assert(N == a_rhs.box().size(2));
  Vector<Real> rhs(N);

  IntVect lo = a_box.smallEnd();
  IntVect hi = a_box.bigEnd();
  for (int j = lo[1]; j <= hi[1]; j++)
    for (int i = lo[0]; i <= hi[0]; i++)
    {
      // Copy rhs k-column into local vector
      for (int k=0; k < N; k++)
        rhs[k] = a_rhs(IntVect(i,j,lo[2] + k)); 

      // Create a copy, because this overwrites it
      LapackFactorization AB;
      AB.define(a_AB);
      LapackWrapper::factorBandMatrix(AB);
      LapackWrapper::solveBandMatrix(AB, rhs.stdVector().data());

      // Copy soln k-column from local vector
      for (int k=0; k < N; k++)
        a_state(IntVect(i,j,lo[2] + k)) = rhs[k]; 
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
