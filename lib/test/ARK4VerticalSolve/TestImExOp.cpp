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

#include "NamespaceHeader.H"

const Real TestImExOp::s_cI = -2;
const Real TestImExOp::s_cE = .1;

void
TestImExOp::resetDt(Real a_dt)
{
  m_dt = a_dt;
}

void
TestImExOp::explicitOp(Real               a_time,
                       std::size_t        a_stage,
                       TestOpData&        a_result,
                       const TestOpData&  a_state)
{
  const LevelData<FArrayBox>& stateData = a_state.data();

  const DisjointBoxLayout& layout = stateData.disjointBoxLayout();
  DataIterator dit = layout.dataIterator();

  LevelData<FArrayBox>& resultData = a_result.data();
  for (dit.begin(); dit.ok(); ++dit)
    {
      const FArrayBox& stateDataFab = stateData[dit];
      FArrayBox& resultDataFab = resultData[dit];

      // This is just a simple test for the time integration
      Real coef = s_cE + 1.0/(1.0 + a_time);
      resultDataFab.copy(stateDataFab);
      resultDataFab.mult(coef);
    }
}

void
TestImExOp::implicitOp(Real               a_time,
                       std::size_t        a_stage,
                       TestOpData&        a_result,
                       const TestOpData&  a_state)
{
  const LevelData<FArrayBox>& stateData = a_state.data();
  const DisjointBoxLayout& grids = stateData.disjointBoxLayout();

  a_result.define(a_state);
  LevelData<FArrayBox>& resultData = a_result.data();

  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      const FArrayBox& stateDataFab = stateData[dit];
      FArrayBox& resultDataFab = resultData[dit];
      resultDataFab.setVal(0.);

      // Just a simple test op for the time integration
      resultDataFab.copy(stateDataFab);
      resultDataFab.mult(s_cI);
    }
}

void
TestImExOp::solve(Real          a_time,
                  std::size_t   a_stage,
                  Real          a_dtscale,
                  TestOpData&   a_soln)
{
  LevelData<FArrayBox>& solnData = a_soln.data();
  const DisjointBoxLayout& grids = solnData.disjointBoxLayout();

  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      FArrayBox& solnDataFab = solnData[dit];

      // Just a simple test op solve for the time integration
      Real coef = 1.0 / (1.0 - a_dtscale*s_cI*m_dt);
      solnDataFab.mult(coef);
    }
}

#include "NamespaceFooter.H"
