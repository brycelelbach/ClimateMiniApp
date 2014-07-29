/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach aka wash <blelbach@cct.lsu.edu>
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/include/iostreams.hpp>

#include "UsingBaseNamespace.H"

#include "HPXDriver.H"

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    hpx::cout << "hello world\n" << hpx::flush;
    return 0;
} 

int main(int argc, char** argv)
{
    // If this doesn't completely blow up, test was succesful.
    return init(chombo_main, argc, argv); // Doesn't return
}

