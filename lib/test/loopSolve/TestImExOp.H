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
#include "TestData.H"
#include "LapackFactorization.H"

#include "NamespaceHeader.H"

class TestImExOp
{
public:
  static const Real s_cI;

  /// Default constructor
  TestImExOp();

  /// Destructor
  virtual ~TestImExOp();

  virtual void define(const TestData&   a_state,
                      Real a_dt,
                      Real a_dtscale);

  virtual bool isDefined() const { return m_isDefined; }

  virtual void resetDt(Real a_dt);

  void exact(TestData& a_exact, Real a_time);

  void implicitSolve(const pair<DataIndex,Box>& a_tile, 
    TestData& a_soln, TestData& a_rhs, Real a_time, Real a_dt);

  void buildTridiagonal(LapackFactorization& a_A,
    Real a_c1, Real a_c2, Real& a_dx, int a_N);

  void implicitSolveTridiag(FArrayBox& a_state, FArrayBox& a_rhs,
    Box a_box, LapackFactorization& a_AB);

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
