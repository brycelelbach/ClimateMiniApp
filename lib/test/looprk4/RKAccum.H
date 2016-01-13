#include "StencilLoopMacros.H"

#ifndef _RKACCUM_H_
#define _RKACCUM_H_

template<int N>
struct RKAccum {
  const int nAccum=N;
  Real scale[N];
  LevelData<FArrayBox>* accum[N];
};

template<int N>
struct RKAccumFAB {
  const int nAccum=N;
  Real scale[N];
  Real* accum[N];
  dataix d;
};

#endif
