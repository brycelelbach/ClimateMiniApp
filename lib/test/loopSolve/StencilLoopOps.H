#ifndef _STENCILLOOPOPS_H_
#define _STENCILLOOPOPS_H_

#include "FArrayBox.H"
#include "LevelData.H"
#include "TestSolnData.H"
#include "TestRhsData.H"
#include "RKAccum.H"
#include "TestImExOp.H"

#include "NamespaceHeader.H"

/**
 * setValLoop
 * Set a cell-centered fab component to the value using the macro SET_OP_IJK
 * a_data - cell-centered fab that will be set
 * a_box - cell-centered box where value will be set
 * val - value to set fab to
 * a_comp - component of the fab to set
 */
#define SET_OP_IJK(val)       \
  (           \
                val \
  )

template<int N> void
setValLoop(FArrayBox& a_data, const Box& a_box, 
    Real a_val, int a_comp, RKAccumFAB<N>& a_rkfab)
{
  CH_assert(a_data.box().contains(a_box));

  CH_TIMERS("TestImExOp::setValLoop");
  CH_TIMER("loop",t);

  loop l;
  SET_LOOP(l,a_box);

  dataix d;
  SET_DATAIX(d,a_data,a_box);
  Real* data = a_data.dataPtr(a_comp);

  CH_START(t);
  int i,j,k,n;
  for (k=0; k<l.size[2]; k++) {
  for (j=0; j<l.size[1]; j++) {
  for (i=0; i<l.size[0]; i++) {
    int ijk = OFFSET_DATAIX(d);
    Real opval = SET_OP_IJK(a_val);
    data[ijk] = opval;

    // update any other pointers with the operator
    int aijk = OFFSET_DATAIX(a_rkfab.d);
    for (n=0; n < N; n++)
      a_rkfab.accum[n][aijk] += a_rkfab.scale[n]*opval;
  } } }
  CH_STOP(t);
}


template<int N> void
setScaleLoop(FArrayBox& a_op, const Box& a_box, 
    const FArrayBox& a_data, Real a_scale, int a_comp, RKAccumFAB<N>& a_rkfab)
{
  CH_assert(a_data.box().contains(a_box));

  CH_TIMERS("setScaleLoop");
  CH_TIMER("loop",t);

  loop l;
  SET_LOOP(l,a_box);

  dataix o;
  SET_DATAIX(o,a_op,a_box);
  Real* op = a_op.dataPtr(a_comp);

  dataix d;
  SET_DATAIX(d,a_data,a_box);
  const Real* data = a_data.dataPtr(a_comp);

  CH_START(t);
  int i,j,k,n;
  for (k=0; k<l.size[2]; k++) {
  for (j=0; j<l.size[1]; j++) {
  for (i=0; i<l.size[0]; i++) {
    int ijk = OFFSET_DATAIX(o);
    int dijk = OFFSET_DATAIX(d);
    Real opval = a_scale*data[dijk];
    op[ijk] = opval;

    // update any other pointers with the operator
    int aijk = OFFSET_DATAIX(a_rkfab.d);
    for (n=0; n < N; n++)
      a_rkfab.accum[n][aijk] += a_rkfab.scale[n]*opval;
  } } }
  CH_STOP(t);
}


template<int N> void
explicitOp(const pair<DataIndex,Box>& a_tile, TestRhsData& a_result, 
    Real a_time, const TestSolnData& a_state, RKAccum<N>& a_rk)
{
  CH_TIMERS("explicitOp");

  DataIndex dataix = a_tile.first;
  const FArrayBox& stateDataFab = a_state.fab(dataix);
  FArrayBox& resultDataFab = a_result.fab(dataix);
  // Box b = a_state.domain(a_dit);
  Box b = a_tile.second;
  // Stuff this box's RK temps into a local loop struct
  RKAccumFAB<N> rkfab;
  for (int n = 0; n < N; n++)
  {
    FArrayBox& fab = (*a_rk.accum[n]).fab(dataix);
    SET_DATAIX(rkfab.d,fab,b); // NOTE: assumes size of SolnData
    rkfab.scale[n] = a_rk.scale[n];
    rkfab.accum[n] = fab.dataPtr(0);
  }
  
  // This is just a simple test for the time integration
  const Real cE = TestImExOp::s_cE;
  Real scale = 1.0*cE/(1.0 + cE*a_time); // d/dt = this * phi
  setScaleLoop<N>(resultDataFab, b, stateDataFab, scale, 0, rkfab);
}


inline void vectorizedCopy(const FArrayBox& a_from, FArrayBox& a_to, Box a_tile)
{
  CH_TIMERS("vectorizedCopy");
  CH_assert(a_from.size() == a_to.size());
  CH_assert(a_from.nComp() == a_to.nComp());

  Real const* from_p = a_from.dataPtr();
  Real* to_p = a_to.dataPtr();
  int const numVals = a_to.nComp()*(a_to.box().numPts());
  #pragma simd
  for (int i=0; i < numVals; i++)
    to_p[i] = from_p[i];
}


void
implicitSolve(const pair<DataIndex,Box>& a_tile, 
    TestSolnData& a_soln, TestRhsData& a_rhs, Real a_time, Real a_dt)
{
  CH_TIMERS("implicitSolve");

  /*
  LevelData<FArrayBox>& solnData = a_soln.data();
  const DisjointBoxLayout& grids = solnData.disjointBoxLayout();
  LevelData<FArrayBox>& rhsData = a_rhs.data();
  DataIterator dit = grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
  {
    FArrayBox& solnDataFab = solnData[dit];
    FArrayBox& rhsDataFab = rhsData[dit];
  */

  DataIndex dataix = a_tile.first;
  FArrayBox& solnDataFab = a_soln.fab(dataix);
  FArrayBox& rhsDataFab = a_rhs.fab(dataix);
  Box b = a_tile.second;

  // Just a simple test op solve for the time integration
  Real cI = TestImExOp::s_cI;
  Real coef = 1.0 / (1.0 - cI*a_dt);
  solnDataFab.copy(rhsDataFab, b);
  solnDataFab.mult(coef, b);

  // Contract is to return the operator in the rhs
  rhsDataFab.copy(solnDataFab, b);
  rhsDataFab.mult(cI, b);

/*
  // This is just a simple test for the time integration
  const Real cE = TestImExOp::s_cE;
  Real scale = 1.0*cE/(1.0 + cE*a_time); // d/dt = this * phi
  setScaleLoop<N>(resultDataFab, b, stateDataFab, scale, 0, rkfab);
*/
}
#include "NamespaceFooter.H"

#endif
