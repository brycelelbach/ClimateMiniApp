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
#include "LapackFactorization.H"

#include "NamespaceHeader.H"
/// A struct to hold the diagonals of a banded matrix
struct band_matrix
{
  const static int maxbands = 7;
  int nbands;
  std::vector<Real> d[maxbands];
};

class TestOperator
{
public:
  TestOperator();
  virtual ~TestOperator();

  static void build4thOrderOperator(band_matrix& a_A, 
      Real& a_coef, Real& a_dx, int a_N);

  static void build2ndOrderOperator(band_matrix& a_A, 
      Real& a_coef, Real& a_dx, int a_N);

  static void build2ndOrderSolver(band_matrix& a_A, 
      Real& a_coef, Real& a_dx, int a_N);

  static void build4thOrderBanded(LapackFactorization& a_AB,
      Real& a_coef, Real& a_dx, int a_N);

  static void applyBandedMatrix(FArrayBox& a_state, 
      FArrayBox& a_rhs, const band_matrix& a_A, Real a_alpha, Real a_beta);

  static void implicitSolve(FArrayBox& a_state,
                            FArrayBox& a_rhs,
                            const band_matrix& a_A);

  static void implicitSolveBanded(FArrayBox& a_state, 
      FArrayBox& a_rhs, LapackFactorization& a_AB);

  static void setExact(FArrayBox& a_exact,
                       int a_kx,
                       Real a_dx,
                       Real a_coef);

};

#include "NamespaceFooter.H"

#endif
