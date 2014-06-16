#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _MPIINITIALIZER_H_
#define _MPIINITIALIZER_H_

#include "DetectCXX11.H"

// RAII pattern.
struct MPIInitializer
{
    MPIInitializer(int argc, char** argv)
    {
        #ifdef MPI
        MPI_Init(&argc, &argv)
        #endif
    }

    ~MPIInitializer()
    {
        #ifdef MPI
        MPI_Finalize();
        #endif
    }

#if !defined(CH_CXX11)
  private:
    MPIInitializer (MPIInitializer const&);
    MPIInitializer& operator= (MPIInitializer const&);
#else
    MPIInitializer (MPIInitializer const&) = delete;
    MPIInitializer& operator= (MPIInitializer const&) = delete;
#endif
};

#endif

