#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _TestImExDxxOp_H_
#define _TestImExDxxOp_H_

#include "FArrayBox.H"
#include "LevelData.H"
#include "TestOpData.H"

#include "NamespaceHeader.H"

// See README for the math

class TestImExDxxOp
{
public:
  static Real s_alpha;
  static Real s_beta;
  static Real s_cE;
  static Real s_cI;
  static Real s_kx;
  static Real s_C;

  /// Default constructor
  TestImExDxxOp();

  /// Destructor
  virtual ~TestImExDxxOp();

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

  static void setCoefs(Real a_alpha, Real a_beta, int a_kx, Real a_C);

  static void setExact(LevelData<FArrayBox>& a_exact, Real a_time);

protected:

  bool m_isDefined;

  Real m_dt;

  Real m_dtscale;

private:
  // Disallowed for all the usual reasons
  void operator=(const TestImExDxxOp&);
  TestImExDxxOp(const TestImExDxxOp&);
};

#include "NamespaceFooter.H"

#endif
