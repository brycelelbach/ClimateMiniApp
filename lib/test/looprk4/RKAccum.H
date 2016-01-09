#include "StencilLoopMacros.H"

#ifndef _RKACCUM_H_
#define _RKACCUM_H_

struct RKAccum {
  int nAccum;
  Real scale[10];
  LevelData<FArrayBox>* accum[10];
};

struct RKAccumFAB {
  int nAccum;
  Real scale[10];
  Real* accum[10];
  dataix d;
  // FArrayBox* accum[10];
};

#endif
