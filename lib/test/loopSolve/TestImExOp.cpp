#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "BoxIterator.H"
#include "TestImExOp.H"
#include "LevelDataOps.H"
#include "LapackFactorization.H"
#include "Lapack.H"

#include "NamespaceHeader.H"

const Real TestImExOp::s_cI = 0.3;

TestImExOp::TestImExOp()
{
  m_isDefined = false;
}

TestImExOp::~TestImExOp()
{
}

void
TestImExOp::define(const TestData&   a_state,
                    Real a_dt,
                    Real a_dtscale)
{
  m_dt = a_dt;
  m_dtscale = a_dtscale;
  m_isDefined = true;
}

void
TestImExOp::exact(TestData& a_exact, Real a_time)
{
  CH_assert(isDefined());
  // Real coef = s_cI + s_cE + s_cS;
  // return exp(coef*a_time);
  // Real coef = s_cI + s_cE + s_cS;
  // return 1.0 + coef*a_time;
  // return  pow(1.0 + s_cI*a_time,1); // (1 + cI*t)^1

  // Calculate the cos() exact solution scaling for this many steps
  LevelData<FArrayBox>& exact = a_exact.data();
  DisjointBoxLayout dbl = exact.disjointBoxLayout();
  Box domain = dbl.physDomain().domainBox();
  RealVect dx = 1.0 / RealVect(domain.size());
  Real kx = 2.0*M_PI;
  Real ky = 2.0*M_PI;
  Real kz = 1.0*M_PI;
  // Calculate the amplification factor for this wave number
  Real gamma = s_cI*2.0*(cos(kz*dx[2]) - 1.0)/(dx[2]*dx[2]);

  // How many steps have we taken?
  Real steps = a_time / m_dt;
  Real exactScale = pow(1.0 - gamma*m_dt,-steps); // (1 / (1 - g*dt))^steps

  DataIterator dit = exact.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
  {
    FArrayBox& exactFab = exact[dit];
    Box b = dbl[dit];
    for (BoxIterator bit(b); bit.ok(); ++bit)
    {
      IntVect iv = bit();
      RealVect xyz = (RealVect(iv) + 0.5)* dx;
      exactFab(iv,0) = exactScale*
        cos(kx*xyz[0])*cos(ky*xyz[1])*cos(kz*xyz[2]);
    }
  }
}


void
TestImExOp::resetDt(Real a_dt)
{
  CH_assert(isDefined());
  m_dt = a_dt;
}


void
TestImExOp::implicitSolve(const std::pair<DataIndex,Box>& a_tile, 
    TestData& a_soln, Real a_time, Real a_dt)
{
  CH_TIMERS("implicitSolve");

  const DisjointBoxLayout& dbl = a_soln.data().disjointBoxLayout();
  Box domain = dbl.physDomain().domainBox();

  DataIndex dataix = a_tile.first;
  FArrayBox& solnDataFab = a_soln.fab(dataix);
  Box b = a_tile.second;

  // Just a simple test op solve for the time integration
  Real cI = TestImExOp::s_cI;
  RealVect dx = 1.0 / RealVect(domain.size());
  int N = domain.size(2);

  // These values are for creating a 2nd-order (I - dt*cI*Dzz)
  Real c1 = 1;
  Real c2 = -cI*a_dt;
  LapackFactorization A;
  buildTridiagonal(A, c1, c2, dx[2], N);
  implicitSolveTridiag(solnDataFab, b, A);
}


/// Creates a 2nd-order tri-diagonal matrix for a backward Euler
/// heat equation step: (c1 * I - c2 * D_xx) with homogeneous Neumann bc's
void 
TestImExOp::
buildTridiagonal(LapackFactorization& a_A,
    Real a_c1, Real a_c2, Real& a_dx, int a_N)
{
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
    }

  // Fix the diagonal ends for the homogeneous Neumann bc's
  // For 2nd-order operator
  const Real bc = -1.0;
  a_A(0,0) = a_c1 + bc*coef;
  a_A(ncol-1,ncol-1) = a_c1 + bc*coef;
}

void tridiagonal_solve_native(
    std::vector<double>& a, // Lower band
    std::vector<double>& b, // Diagonal
    std::vector<double>& c, // Upper band
    std::vector<double>& u  // Solution
    )
{
    auto const nx = u.size();

    for (int i = 1; i < nx; i++)
    {
        double m = a[i-1]/b[i-1];
        b[i] -= m*c[i-1];
        u[i] -= m*u[i-1];
    }

    // solve for last x value
    u[nx-1] = u[nx-1]/b[nx-1];
 
    // solve for remaining x values by back substitution
    for(int i = nx - 2; i >= 0; i--)
        u[i] = (u[i] - c[i]*u[i+1])/b[i];
}

/// Does a tridiagonal solve for a 2nd-order backward Euler
/// heat equation step: (I - coef * D_zz) state = rhs
void 
TestImExOp::
implicitSolveTridiag(
    FArrayBox& a_state,
    Box a_box,
    LapackFactorization& a_AB)
{
  CH_TIME("TestImExOp::implicitSolveTridiag");

  int N = a_box.size(2); // z-direction stencil
  assert(0 == a_box.smallEnd()[2]); // no ghost zones in the z-direction
  assert(N == a_state.box().size(2));
  Vector<Real> rhs(N);
  // Tri-diag matrix diagonals
  Vector<Real> lower(N);
  Vector<Real> diag(N);
  Vector<Real> upper(N);

  IntVect lo = a_box.smallEnd();
  IntVect hi = a_box.bigEnd();
  for (int j = lo[1]; j <= hi[1]; j++)
    for (int i = lo[0]; i <= hi[0]; i++)
    {
      // Copy rhs k-column into local vector
      for (int k=0; k < N; k++)
        rhs[k] = a_state(IntVect(i,j,lo[2] + k)); 

      // LAPACK overwrites things, so need to copy
      for (int k=0; k < N; k++)
        diag[k] = a_AB(k,k); // k row k col
      for (int k=0; k < N-1; k++)
        lower[k] = a_AB(k+1,k); // k+1 row k col
      for (int k=0; k < N-1; k++)
        upper[k] = a_AB(k,k+1); // k row k+1 col

/*
      // Sorry, need these for the fortran version of LAPACK
      int info = 0;
      int NRHS = 1;
      int LDB = N;
      dgtsv_(
          // LAPACK_ROW_MAJOR, // matrix format
          &N, // matrix order
          &NRHS, // # of right hand sides 
          lower.stdVector().data() + 1, // 1st subdiagonal part
          diag.stdVector().data(), // diagonal part
          upper.stdVector().data(), // 1st superdiagonal part
          rhs.stdVector().data(), // column to solve 
          &LDB, // leading dimension of RHS
          &info
          );

      if (info != 0)
        pout() << "Lapack info=" << info << endl;
      assert(info == 0);
*/

      tridiagonal_solve_native(
        lower.stdVector(),
        diag.stdVector(),
        upper.stdVector(),
        rhs.stdVector()
      ); 

      // Copy soln k-column from local vector
      for (int k=0; k < N; k++)
        a_state(IntVect(i,j,lo[2] + k)) = rhs[k]; 
    }
}

#include "NamespaceFooter.H"
