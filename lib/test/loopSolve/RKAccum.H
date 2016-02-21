#include "StencilLoopMacros.H"
#include "TestSolnData.H"

#ifndef _RKACCUM_H_
#define _RKACCUM_H_

template<int N>
struct RKAccum {
  static const int nAccum=N;
  Real scale[N];
  TestSolnData* accum[N];
};

template<int N>
struct RKAccumFAB {
  static const int nAccum=N;
  Real scale[N];
  Real* accum[N];
  dataix d;
};

#endif
