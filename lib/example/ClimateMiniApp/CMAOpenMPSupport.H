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

#if !defined(CHOMBO_26089411_1EE3_407D_B047_0744F161D344)
#define CHOMBO_26089411_1EE3_407D_B047_0744F161D344

#include <boost/preprocessor/stringize.hpp>

#if !defined(CH_HPX)
    #include <omp.h>
#endif

#define CMA_PRAGMA(args) _Pragma(BOOST_PP_STRINGIZE(args))

#if defined(CH_HPX)
    #define CMA_PRAGMA_OMP(args)
#else 
    #define CMA_PRAGMA_OMP(args) CMA_PRAGMA(omp args)
#endif

#endif // CHOMBO_26089411_1EE3_407D_B047_0744F161D344
