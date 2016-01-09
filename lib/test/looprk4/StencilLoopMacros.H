#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _STENCILLOOPMACROS_H_
#define _STENCILLOOPMACROS_H_

#include "IntVect.H"

#include "NamespaceHeader.H"

///
/**
  This is a header file with all the loop structs
  and macros for all the C-style loop operator functions
*/

// Struct data for loops
struct loop
{
  int size[SpaceDim];
  int type[SpaceDim];
  int offset[SpaceDim];
};
/**
 * b - a Box
 */

//    loop.size = b.size().getVect();

#define SET_LOOP(loop,b)       \
  { \
    D_TERM(loop.size[0] = b.size()[0];, \
           loop.size[1] = b.size()[1];, \
           loop.size[2] = b.size()[2];); \
    D_TERM(loop.type[0] = b.type()[0];, \
           loop.type[1] = b.type()[1];, \
           loop.type[2] = b.type()[2];); \
    D_TERM(loop.offset[0] = b.loVect()[0];, \
           loop.offset[1] = b.loVect()[1];, \
           loop.offset[2] = b.loVect()[2];); \
  }

// Struct data for loops indexing into fab data
struct dataix
{
  int size[SpaceDim];
  int stride[SpaceDim];
  int shift[SpaceDim];
};
/**
 * f - an FArrayBox
 * b - a Box
 */
//    dataix.size = f.box().size().getVect();
#define SET_DATAIX(dataix,f,b)       \
  { \
    D_TERM(dataix.size[0] = f.size()[0];, \
           dataix.size[1] = f.size()[1];, \
           dataix.size[2] = f.size()[2];); \
    D_TERM(dataix.shift[0] = b.smallEnd()[0] - f.smallEnd()[0];, \
           dataix.shift[1] = b.smallEnd()[1] - f.smallEnd()[1];, \
           dataix.shift[2] = b.smallEnd()[2] - f.smallEnd()[2];); \
    D_TERM(dataix.stride[0] = 1;, \
           dataix.stride[1] = dataix.size[0];, \
           dataix.stride[2] = dataix.size[0]*dataix.size[1];); \
  }

#define OFFSET_DATAIX(dataix)       \
  ( \
    D_TERM((i+dataix.shift[0])*dataix.stride[0], \
      + (j+dataix.shift[1])*dataix.stride[1], \
      + (k+dataix.shift[2])*dataix.stride[2]) \
  )

#define OFFSET_DATAIX_II(dataix,i,j,k)       \
  ( \
    D_TERM((i+dataix.shift[0])*dataix.stride[0], \
      + (j+dataix.shift[1])*dataix.stride[1], \
      + (k+dataix.shift[2])*dataix.stride[2]) \
  )


#include "NamespaceFooter.H"
#endif
