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
}

void
TestImExOp::exact(TestData& a_exact, Real a_time)
{
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

  // Just a simple test op solve for the time integration
  Real cI = TestImExOp::s_cI;
  RealVect dx = 1.0 / RealVect(domain.size());

  // These values are for creating a 2nd-order (I - dt*cI*Dzz)
  Real c1 = 1;
  Real c2 = -cI*a_dt;

  DataIndex dataix = a_tile.first;
  FArrayBox& A = a_soln.A(dataix);
  FArrayBox& B = a_soln.B(dataix);
  FArrayBox& C = a_soln.C(dataix);
  FArrayBox& U = a_soln.data(dataix);
  Box b = a_tile.second;

  buildTridiagonal(A, B, C, b, c1, c2, dx[2]);
  implicitSolveTridiag(A, B, C, U, b);
}

void 
TestImExOp::
buildTridiagonal(FArrayBox& a_A, FArrayBox& a_B, FArrayBox& a_C,
    Box a_box, Real a_c1, Real a_c2, Real a_dx)
{
  Real coef = a_c2/(1.0*a_dx*a_dx); // for 2nd-order

//  a_A.setVal(0);
//  a_C.setVal(0);

  IntVect const lo = a_box.smallEnd();
  IntVect const hi = a_box.bigEnd();

  int const N = a_box.size()[2];

  for (int j = lo[1]; j <= hi[1]; j++)
    for (int i = lo[0]; i <= hi[0]; i++)
    {
      // Copy rhs k-column into local vector
      for (int k=lo[2]; k <= hi[2]; k++)
      {
        a_B(IntVect(i,j,k)) = a_c1 - 2.0*coef;
      }

      for (int k=lo[2]; k <= hi[2] - 1; k++)
      {
        a_A(IntVect(i,j,k)) = 1.0*coef;
        a_C(IntVect(i,j,k)) = 1.0*coef;
      }

      // Fix the diagonal ends for the homogeneous Neumann bc's
      // For 2nd-order operator
      a_B(IntVect(i,j,lo[2])) = a_c1 - 1.0*coef;
      a_B(IntVect(i,j,hi[2])) = a_c1 - 1.0*coef;
    }

/*
    for (int k=lo[2]; k <= hi[2]; k++)
      std::cout << "A(" << lo[0] << ", " << lo[1] << ", " << k << ") == " << a_A(IntVect(lo[0],lo[1],k)) << "\n";

    for (int k=lo[2]; k <= hi[2]; k++)
      std::cout << "B(" << lo[0] << ", " << lo[1] << ", " << k << ") == " << a_B(IntVect(lo[0],lo[1],k)) << "\n";

    for (int k=lo[2]; k <= hi[2]; k++)
      std::cout << "C(" << lo[0] << ", " << lo[1] << ", " << k << ") == " << a_C(IntVect(lo[0],lo[1],k)) << "\n";
*/
}

void tridiagonal_solve_native(
    FArrayBox& a,
    FArrayBox& b,
    FArrayBox& c,
    FArrayBox& u,
    Box box
    )
{
    IntVect const lo = box.smallEnd();
    IntVect const hi = box.bigEnd();

    auto const nx = box.size()[0];
    auto const ny = box.size()[1];
    auto const nz = box.size()[2];

//    std::cout << "lo[2] == " << lo[2] << "\n";
//    std::cout << "hi[2] == " << hi[2] << "\n";

    // Forward elimination. 
    for (int k = lo[2] + 1; k <= hi[2]; ++k)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* up     = &u(IntVect(lo[0], j, k));
            Real* usub1p = &u(IntVect(lo[0], j, k - 1));

            Real* asub1p = &a(IntVect(lo[0], j, k - 1));

            Real* bp     = &b(IntVect(lo[0], j, k));
            Real* bsub1p = &b(IntVect(lo[0], j, k - 1));

            Real* csub1p = &c(IntVect(lo[0], j, k - 1));

            __assume_aligned(up, 64);
            __assume_aligned(usub1p, 64);

            __assume_aligned(asub1p, 64);

            __assume_aligned(bp, 64);
            __assume_aligned(bsub1p, 64);

            __assume_aligned(csub1p, 64);

            #pragma simd
            for (int i = 0; i <= hi[0] - lo[0]; ++i)
            {
//                std::cout << "asub1p(" << (lo[0] + i) << ", " << j << ", " << k << ") == " << asub1p[i] << "\n";
//                std::cout << "bsub1p(" << (lo[0] + i) << ", " << j << ", " << k << ") == " << bsub1p[i] << "\n";

                // double const m = a[k - 1] / b[k - 1];
                Real const m = asub1p[i] / bsub1p[i];
                // b[k] -= m * c[k - 1];
                bp[i] -= m * csub1p[i];
                // u[k] -= m * u[k - 1];
                up[i] -= m * usub1p[i];
            }
        }

    for (int j = lo[1]; j <= hi[1]; ++j)
    {
        Real* uendp = &u(IntVect(lo[0], j, hi[2]));

        Real* bendp = &b(IntVect(lo[0], j, hi[2]));

        __assume_aligned(uendp, 64);

        __assume_aligned(bendp, 64);

        for (int i = 0; i <= hi[0] - lo[0]; ++i)
        {
            // u[nz - 1] = u[nz - 1] / b[nz - 1];
            uendp[i] = uendp[i] / bendp[i];
        }
    }
 
    // Back substitution. 
    for (int k = hi[2] - 1; k >= lo[2]; --k)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* up     = &u(IntVect(lo[0], j, k));
            Real* uadd1p = &u(IntVect(lo[0], j, k + 1));

            Real* bp     = &b(IntVect(lo[0], j, k));

            Real* cp     = &c(IntVect(lo[0], j, k));

            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            #pragma simd
            for (int i = 0; i <= hi[0] - lo[0]; ++i)
            {
                // u[k] = (u[k] - c[k] * u[k + 1]) / b[k];
                up[i] = (up[i] - cp[i] * uadd1p[i]) / bp[i];
            }
        }
}

void 
TestImExOp::
implicitSolveTridiag(
    FArrayBox& a_A,
    FArrayBox& a_B,
    FArrayBox& a_C,
    FArrayBox& a_U,
    Box a_box)
{
  tridiagonal_solve_native(a_A, a_B, a_C, a_U, a_box);

/*
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

#if 0
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
#endif

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
*/
}

#include "NamespaceFooter.H"
