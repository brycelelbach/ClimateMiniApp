#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

#ifndef _LAPACKWRAPPER_H_
#define _LAPACKWRAPPER_H_

#include "FArrayBox.H"
#include "LapackFactorization.H"
#include "MayDay.H"

#include "NamespaceHeader.H"

///
/**
    Static class to wrap lapack calls
 */
class LapackWrapper
{
public:

  static void applyBandMatrix(Real* in, Real* out, 
      LapackFactorization& A);

  static void factorBandMatrix(LapackFactorization& A);

  static void solveBandMatrix(LapackFactorization& A, Real* inout);

protected:

private:
  LapackWrapper();
  ~LapackWrapper();
  LapackWrapper(const LapackWrapper&);
  void operator=(const LapackWrapper&);
};

#include "NamespaceFooter.H"
#endif
