#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "TestImExDxxOp.H"
#include "TestOperator.H"

#include "NamespaceHeader.H"
Real TestImExDxxOp::s_alpha;
Real TestImExDxxOp::s_beta;
Real TestImExDxxOp::s_cI;
Real TestImExDxxOp::s_cE;
Real TestImExDxxOp::s_kx;
Real TestImExDxxOp::s_C;

TestImExDxxOp::TestImExDxxOp()
{
  m_isDefined = false;
}

TestImExDxxOp::~TestImExDxxOp()
{
}

void
TestImExDxxOp::setCoefs(Real a_alpha, Real a_beta, int a_kx, Real a_C)
{
  s_alpha = a_alpha;
  s_beta = a_beta;
  s_kx = a_kx;
  s_C = a_C;
  // s_cI = - a_alpha * a_beta;
  // s_cE = - a_alpha * (1.0 - a_beta);
  s_cI = a_alpha * a_beta;
  // s_cI = - a_alpha * pow(M_PI*s_kx,2) * a_beta;
  s_cE = - a_alpha * pow(M_PI*s_kx,2) * (1.0 - a_beta);
}

void
TestImExDxxOp::define(const TestOpData& a_state, Real a_dt, Real a_dtscale)
{
  m_dt = a_dt;
  m_dtscale = a_dtscale;
  m_isDefined = true;
}

void
TestImExDxxOp::resetDt(Real a_dt)
{
  CH_assert(isDefined());
  m_dt = a_dt;
}

void
TestImExDxxOp::explicitOp(TestOpData& a_result, Real a_time, 
    const TestOpData&  a_state)
{
  CH_assert(isDefined());

  const LevelData<FArrayBox>& stateData = a_state.data();

  const DisjointBoxLayout& layout = stateData.disjointBoxLayout();
  DataIterator dit = layout.dataIterator();

  LevelData<FArrayBox>& resultData = a_result.data();
  for (dit.begin(); dit.ok(); ++dit)
    {
      const FArrayBox& stateDataFab = stateData[dit];
      FArrayBox& resultDataFab = resultData[dit];

      // Explicit operator is s_cE * \phi
      Real coef = s_cE;
      resultDataFab.copy(stateDataFab);
      resultDataFab.mult(coef);
    }
}

void
TestImExDxxOp::implicitOp(TestOpData&             a_result,
                        Real                    a_time,
                        const TestOpData&  a_state)
{
  CH_assert(isDefined());

  const LevelData<FArrayBox>& stateData = a_state.data();
  const DisjointBoxLayout& grids = stateData.disjointBoxLayout();

  a_result.define(a_state);
  LevelData<FArrayBox>& resultData = a_result.data();

  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
  {
    FArrayBox& stateDataFab = const_cast<FArrayBox&>(stateData[dit]);
    FArrayBox& resultDataFab = resultData[dit];
    resultDataFab.setVal(0.);
    // Just a simple test op for the time integration
    /*
    resultDataFab.copy(stateDataFab);
    resultDataFab.mult(s_cI);
    */

    int N = grids[dit].size(0); // number of points in x dir
    Real dx = 1.0 / (Real) N; // FIXME - not always true?
    LapackFactorization A;
    // TestOperator::build4thOrderBanded(A, s_cI, 0.0, dx, N);
    TestOperator::build4thOrderBanded(A, 0.0, s_cI, dx, N);
    TestOperator::applyBandedMatrix(stateDataFab, resultDataFab,  A);
  }
}

void
TestImExDxxOp::solve(TestOpData&   a_soln,
                   const TestOpData&  a_rhs,
                   Real               a_time)
{
  CH_assert(isDefined());

  LevelData<FArrayBox>& solnData = a_soln.data();
  const DisjointBoxLayout& grids = solnData.disjointBoxLayout();
  const LevelData<FArrayBox>& rhsData = a_rhs.data();

  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
  {
    FArrayBox& solnDataFab = solnData[dit];
    FArrayBox& rhsDataFab = const_cast<FArrayBox&>(rhsData[dit]);

    // Just a simple test op solve for the time integration
    /*
    Real coef = 1.0 / (1.0 - m_dtscale*s_cI*m_dt);
    solnDataFab.copy(rhsDataFab);
    solnDataFab.mult(coef);
    */

    int N = grids[dit].size(0); // number of points in x dir
    Real dx = 1.0 / (Real) N; // FIXME - not always true?
    LapackFactorization A;
    // Real coef = 1.0 - m_dtscale*s_cI*m_dt;
    // TestOperator::build4thOrderBanded(A, coef, 0.0, dx, N);
    Real coef = - m_dtscale*s_cI*m_dt;
    TestOperator::build4thOrderBanded(A, 1.0, coef, dx, N);
    TestOperator::implicitSolveBanded(solnDataFab, rhsDataFab,  A);
  }
}

void
TestImExDxxOp::setExact(LevelData<FArrayBox>& a_exact, Real a_time)
{
  // Real expo = -s_alpha;
  Real expo = -s_alpha * pow(M_PI*s_kx,2);
  Real decay = exp(expo*a_time)*s_C;

  DataIterator dit = a_exact.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
  {
    FArrayBox& exact = a_exact[dit];
    const IntVect size = exact.box().size();
    const int size0 = size[0]; // x direction size
    const Real dx = 1.0 / (Real) size0; // FIXME - not always true?
    const Real kxpi = M_PI*s_kx;

    const IntVect lower = exact.box().smallEnd();
    for (int j = lower[1]; j < size[1]; j++)
      for (int k = lower[2]; k < size[2]; k++)
      {
        Real* ptr = &exact(IntVect(lower[0], j, k)); 
        for (int i=0; i < size0; i++)
        {
          // Calculate the average of cos(kx pi x)
          Real xhi = dx*(Real)(i+1);
          Real xlo = dx*(Real)i;
          ptr[i] = decay * ((kxpi == 0) ? 1
            : (sin(kxpi*xhi) - sin(kxpi*xlo))/(kxpi*dx));
        }
      }
  }
}

#include "NamespaceFooter.H"
