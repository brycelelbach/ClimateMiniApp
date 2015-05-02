#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _TESTIMEXOP_H_
#define _TESTIMEXOP_H_

#include "FArrayBox.H"
#include "LevelData.H"
#include "TestOpData.H"

#include "NamespaceHeader.H"

// Exact solution is exp((cE + cI)*t)*(1+t)
// So that:
//   explicit op = (1/(1+t) + cE) * phi;
//   implicit op = cI * phi;

class TestImExOp
{
public:
  static const Real s_cE;
  static const Real s_cI;

  /// Default constructor
  TestImExOp(Real a_dt)
    : m_dt(a_dt)
  {}

  void resetDt(Real a_dt);

  void explicitOp(Real               a_time,
                  std::size_t        a_stage,
                  TestOpData&        a_result,
                  const TestOpData&  a_state);

  void implicitOp(Real               a_time,
                  std::size_t        a_stage,
                  TestOpData&        a_result,
                  const TestOpData&  a_state);

  void solve(Real          a_time,
             std::size_t   a_stage,
             Real          a_dtscale,
             TestOpData&   a_soln);

protected:
  Real m_dt;
};

#include "NamespaceFooter.H"

#endif
