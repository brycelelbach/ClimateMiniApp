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
  TestImExOp();

  /// Destructor
  virtual ~TestImExOp();

  virtual void define(const TestOpData&   a_state,
                      Real a_dt,
                      Real a_dtscale);

  virtual bool isDefined() const { return m_isDefined; }

  virtual void resetDt(Real a_dt);

  virtual void explicitOp(TestOpData&             a_result,
                          Real                    a_time,
                          const TestOpData&  a_state);

  virtual void implicitOp(TestOpData&             a_result,
                          Real                    a_time,
                          const TestOpData&  a_state);

  virtual void solve(TestOpData&   a_soln,
                     const TestOpData&   a_rhs,
                     Real               a_time);

protected:

  bool m_isDefined;

  Real m_dt;

  Real m_dtscale;

private:
  // Disallowed for all the usual reasons
  void operator=(const TestImExOp&);
  TestImExOp(const TestImExOp&);
};

#include "NamespaceFooter.H"

#endif
