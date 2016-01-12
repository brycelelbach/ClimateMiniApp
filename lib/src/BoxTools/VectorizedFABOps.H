/*
 *      _______              __ 
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Vectorized implementations of operations on FArrayBox-like types of identical
// size and shape (e.g. their Boxes compare equal).

#if !defined(CHOMBO_9FDD09ED_F552_410B_B996_ADFF7803B4CD)
#define CHOMBO_9FDD09ED_F552_410B_B996_ADFF7803B4CD

#include <cstdint>
#include <type_traits>

#include "CH_assert.H"

#include "NamespaceHeader.H"

template <typename F, typename FAB, typename... Args>
inline void vectorizedPointToPointOp(
    FAB const& from
  , int fromComp
  , FAB& to
  , int toComp
  , Args&&... args
    )
{ 
    CH_assert(to.box() == from.box());
    CH_assert(fromComp < from.nComp());
    CH_assert(toComp < to.nComp());

    Real const* __restrict__ from_p = from.dataPtr(fromComp);
    Real* __restrict__       to_p   = to.dataPtr(toComp);

    // Verify that we do not, in fact, have an aliasing issue.
    CH_assert(to_p != from_p);

    IntVect const lower = to.smallEnd();
    IntVect const upper = to.bigEnd(); 

    std::ptrdiff_t const size = (upper[0]-lower[0]+1)
                              * (upper[1]-lower[1]+1)
                              * (upper[2]-lower[2]+1);

    #pragma simd
    for (std::ptrdiff_t i = 0; i < size; ++i)
        F::call(from_p[i], to_p[i], std::forward<Args>(args)...); 
}

template <typename F, typename FAB, typename... Args>
inline void vectorizedScalarToPointOp(
    FAB& to
  , int toComp
  , Args&&... args
    )
{ 
    CH_assert(toComp < to.nComp());

    Real* __restrict__ to_p = to.dataPtr(toComp);

    IntVect const lower = to.smallEnd();
    IntVect const upper = to.bigEnd(); 

    std::ptrdiff_t const size = (upper[0]-lower[0]+1)
                              * (upper[1]-lower[1]+1)
                              * (upper[2]-lower[2]+1);

    #pragma simd
    for (std::ptrdiff_t i = 0; i < size; ++i)
        F::call(to_p[i], std::forward<Args>(args)...); 
}

struct VectorizedCopyKernel
{
    template <typename Value>
    static void call(Value const& from, Value& to)
    {
        to = from;
    }
};

struct VectorizedPointToPointPlusKernel
{
    template <typename Value>
    static void call(Value const& from, Value& to)
    {
        to += from;
    }

    template <typename Value, typename Factor>
    static void call(Value const& from, Value& to, Factor factor)
    {
        to += factor * from;
    }
};

struct VectorizedScalarToPointPlusKernel
{
    template <typename Value, typename Scalar>
    static void call(Value& to, Scalar scalar)
    {
        to += scalar;
    }
};

template <typename FAB, typename Scalar>
inline typename std::enable_if<std::is_arithmetic<Scalar>::value>::type
vectorizedPlus(
    FAB& to
  , Scalar scalar
    )
{ 
    typedef VectorizedScalarToPointPlusKernel kernel;
    for (int n = 0; n < to.nComp(); ++n)
        vectorizedScalarToPointOp<kernel>(to, n, scalar); 
}

template <typename FAB, typename Scalar>
inline typename std::enable_if<std::is_arithmetic<Scalar>::value>::type
vectorizedPlus(
    FAB& to
  , int toComp
  , Scalar scalar
    )
{ 
    typedef VectorizedScalarToPointPlusKernel kernel;
    vectorizedScalarToPointOp<kernel>(to, toComp, scalar); 
}

template <typename FAB>
inline void vectorizedPlus(
    FAB const& from
  , FAB& to
    )
{ 
    typedef VectorizedPointToPointPlusKernel kernel;
    for (int n = 0; n < to.nComp(); ++n)
        vectorizedPointToPointOp<kernel>(from, n, to, n); 
}

template <typename FAB, typename Factor>
inline void vectorizedPlus(
    FAB const& from
  , FAB& to
  , Factor factor
    )
{ 
    typedef VectorizedPointToPointPlusKernel kernel;
    for (int n = 0; n < to.nComp(); ++n)
        vectorizedPointToPointOp<kernel>(from, n, to, n, factor); 
}

template <typename FAB>
inline void vectorizedPlus(
    FAB const& from
  , int fromComp
  , FAB& to
  , int toComp
    )
{ 
    typedef VectorizedPointToPointPlusKernel kernel;
    vectorizedPointToPointOp<kernel>(from, fromComp, to, toComp); 
}

template <typename FAB, typename Factor>
inline void vectorizedPlus(
    FAB const& from
  , int fromComp
  , FAB& to
  , int toComp
  , Factor factor
    )
{ 
    typedef VectorizedPointToPointPlusKernel kernel;
    vectorizedPointToPointOp<kernel>(from, fromComp, to, toComp, factor); 
}

template <typename FAB>
inline void vectorizedCopy(
    FAB const& from
  , FAB& to
    )
{ 
    typedef VectorizedCopyKernel kernel;
    for (int n = 0; n < to.nComp(); ++n)
        vectorizedPointToPointOp<kernel>(from, n, to, n); 
}

template <typename FAB>
inline void vectorizedCopy(
    FAB const& from
  , int fromComp
  , FAB& to
  , int toComp
    )
{ 
    typedef VectorizedCopyKernel kernel;
    vectorizedPointToPointOp<kernel>(from, fromComp, to, toComp); 
}

#include "NamespaceFooter.H"

#endif // CHOMBO_9FDD09ED_F552_410B_B996_ADFF7803B4CD

