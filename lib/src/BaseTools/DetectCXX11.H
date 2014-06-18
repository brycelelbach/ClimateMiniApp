#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _DETECTCXX11_H_
#define _DETECTCXX11_H_

//  (C) Copyright Louisiana State University, 2008-2014
//  (C) Copyright John Maddock 2001-8.
//  (C) Copyright Peter Dimov 2001.
//  (C) Copyright Jens Maurer 2001.
//  (C) Copyright David Abrahams 2002 - 2003.
//  (C) Copyright Aleksey Gurtovoy 2002 - 2003.
//  (C) Copyright Guillaume Melquiond 2002 - 2003.
//  (C) Copyright Beman Dawes 2003.
//  (C) Copyright Martin Wille 2003.
//
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// 
//  Detection code for C++11, for HPX compatibility/C++11 memory model/thread safety stuff. 

#if (__INTEL_COMPILER > 1310) && defined(__GXX_EXPERIMENTAL_CXX0X__)
    #define CH_CXX11 1
#endif

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) && defined(__GXX_EXPERIMENTAL_CXX0X__)
    #define CH_CXX11 1
#endif

#if CH_CXX11
  #define CH_NOEXCEPT noexcept
#else
  #define CH_NOEXCEPT
#endif

#endif
