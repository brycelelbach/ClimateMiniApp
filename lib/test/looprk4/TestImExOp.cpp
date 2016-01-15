#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "TestImExOp.H"
#include "LevelDataOps.H"
#include "StencilLoopMacros.H"
#include "StencilLoopOps.H"

#include "NamespaceHeader.H"

// const Real TestImExOp::s_cI = -0.003;
const Real TestImExOp::s_cI = 0.0;
const Real TestImExOp::s_cE = -0.3;
// const Real TestImExOp::s_cS = -0.001;
const Real TestImExOp::s_cS = 0.0;

TestImExOp::TestImExOp()
{
  m_isDefined = false;
}

TestImExOp::~TestImExOp()
{
}

void
TestImExOp::define(const TestSolnData&   a_state,
                    Real a_dt,
                    Real a_dtscale)
{
  m_dt = a_dt;
  m_dtscale = a_dtscale;
  m_isDefined = true;
}

Real
TestImExOp::exact(Real a_time)
{
  CH_assert(isDefined());
  // Real coef = s_cI + s_cE + s_cS;
  // return exp(coef*a_time);
  // Real coef = s_cI + s_cE + s_cS;
  // return 1.0 + coef*a_time;
  return  pow(1.0 + s_cE*a_time,1); // (1 + cE*t)^4
}


void
TestImExOp::resetDt(Real a_dt)
{
  CH_assert(isDefined());
  m_dt = a_dt;
}


#if 0
void
TestImExOp::explicitOp(TestRhsData& a_result, Real a_time, 
                       const TestSolnData&  a_state, 
                       RKAccum<N>& a_rk)
{
  CH_TIMERS("TestImExOp::explicitOp");
  CH_assert(isDefined());

  const LevelData<FArrayBox>& stateData = a_state.data();
  LevelData<FArrayBox>& resultData = a_result.data();
  const DisjointBoxLayout& layout = stateData.disjointBoxLayout();
  DataIterator dit = layout.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      const FArrayBox& stateDataFab = stateData[dit];
      FArrayBox& resultDataFab = resultData[dit];
      Box b = layout[dit];
      // Stuff this box's RK temps into a local loop struct
      RKAccumFAB rkfab;
      rkfab.nAccum = a_rk.nAccum;
      for (int n = 0; n < rkfab.nAccum; n++)
      {
        FArrayBox& fab = (*a_rk.accum[n])[dit];
        SET_DATAIX(rkfab.d,fab,b); // NOTE: assumes size of SolnData
        rkfab.scale[n] = a_rk.scale[n];
        rkfab.accum[n] = fab.dataPtr(0);
      }
      
      // This is just a simple test for the time integration
      Real opval = 4.0*s_cE*pow(a_time,3); // d/dt of 1 + cE*t^4
      setValLoop<2>(resultDataFab, b, opval, 0, rkfab);
    }
}
#endif

void
TestImExOp::implicitOp(TestRhsData&             a_result,
                       Real                    a_time,
                       const TestSolnData&  a_state)
{
  CH_TIMERS("TestImExOp::implicitOp");
  CH_assert(isDefined());

  const LevelData<FArrayBox>& stateData = a_state.data();
  const DisjointBoxLayout& grids = stateData.disjointBoxLayout();

  a_result.define(a_state);
  LevelData<FArrayBox>& resultData = a_result.data();

  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      const FArrayBox& stateDataFab = stateData[dit];
      FArrayBox& resultDataFab = resultData[dit];
      // resultDataFab.setVal(0.);

      // Just a simple test op for the time integration
      // resultDataFab.copy(stateDataFab);
      // resultDataFab.mult(s_cI);
      resultDataFab.setVal(s_cI);
    }
}

void
TestImExOp::solve(TestSolnData&   a_soln,
                   TestRhsData&  a_rhs,
                   Real               a_time)
{
  CH_TIMERS("TestImExOp::solve");
  CH_assert(isDefined());

  LevelData<FArrayBox>& solnData = a_soln.data();
  const DisjointBoxLayout& grids = solnData.disjointBoxLayout();
  LevelData<FArrayBox>& rhsData = a_rhs.data();

  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      FArrayBox& solnDataFab = solnData[dit];
      FArrayBox& rhsDataFab = rhsData[dit];

      // Just a simple test op solve for the time integration
      // Real coef = 1.0 / (1.0 - m_dtscale*s_cI*m_dt);
      solnDataFab.copy(rhsDataFab);
      // solnDataFab.mult(coef);
      Real coef = m_dtscale*s_cI*m_dt;
      solnDataFab.plus(coef);

      // Contract is to return the operator in the rhs
      // rhsDataFab.copy(solnDataFab);
      // rhsDataFab.mult(s_cI);
      rhsDataFab.setVal(s_cI);
    }
}


void
TestImExOp::splitSourceOp(TestRhsData&  a_rhs,
    Real a_time, TestSolnData&   a_soln)
{
  CH_TIMERS("TestImExOp::splitSourceOp");
  CH_assert(isDefined());

  LevelData<FArrayBox>& solnData = a_soln.data();
  const DisjointBoxLayout& layout = solnData.disjointBoxLayout();
  DataIterator dit = layout.dataIterator();

  LevelData<FArrayBox>& rhsData = a_rhs.data();
  for (dit.begin(); dit.ok(); ++dit)
    {
      const FArrayBox& stateDataFab = solnData[dit];
      FArrayBox& rhsDataFab = rhsData[dit];

      // This is just a simple test for the split explicit term
      // rhsDataFab.copy(stateDataFab);
      // rhsDataFab.mult(s_cS);
      rhsDataFab.setVal(s_cS);
    }
}

#include "NamespaceFooter.H"
