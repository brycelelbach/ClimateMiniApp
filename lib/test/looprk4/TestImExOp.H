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
#include "TestSolnData.H"
#include "TestRhsData.H"
// #include "RKAccum.H"

#include "NamespaceHeader.H"

// Exact solution is exp((cE + cI + cS)*t)
// So that:
//   explicit op = cE * phi;
//   implicit op = cI * phi;
//   split explicit op = cS * phi;

class TestImExOp
{
public:
  static const Real s_cE;
  static const Real s_cI;
  static const Real s_cS;

  /// Default constructor
  TestImExOp();

  /// Destructor
  virtual ~TestImExOp();

  virtual void define(const TestSolnData&   a_state,
                      Real a_dt,
                      Real a_dtscale);

  virtual bool isDefined() const { return m_isDefined; }

  virtual void resetDt(Real a_dt);

  Real exact(Real a_time);

/*
  virtual void explicitOp(TestRhsData&             a_result,
                          Real                    a_time,
                          const TestSolnData&  a_state, 
                          RKAccum<N>& a_rk4);
*/

  virtual void implicitOp(TestRhsData&             a_result,
                          Real                    a_time,
                          const TestSolnData&  a_state);

  virtual void solve(TestSolnData&   a_soln,
                     TestRhsData&   a_rhs,
                     Real               a_time);

  virtual void splitSourceOp(TestRhsData& a_rhs,
                             Real a_time,
                             TestSolnData&   a_soln);

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
