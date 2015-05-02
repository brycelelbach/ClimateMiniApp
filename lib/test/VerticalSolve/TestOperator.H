#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _TESTOPERATOR_H_
#define _TESTOPERATOR_H_

#include "FArrayBox.H"

#include "NamespaceHeader.H"

class TestOperator
{
public:
  TestOperator();
  virtual ~TestOperator();

  static void implicitSolve2ndOrder(FArrayBox& a_state,
                                    FArrayBox& a_rhs,
                                    Real a_coef);

  static void implicitSolve4thOrder(FArrayBox& a_state,
                                    FArrayBox& a_rhs,
                                    Real a_coef);


  static void setExact(FArrayBox& a_exact,
                       int a_kx,
                       Real a_coef);

};

#include "NamespaceFooter.H"

#endif
