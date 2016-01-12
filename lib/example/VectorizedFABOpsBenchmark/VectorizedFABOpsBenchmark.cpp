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

#include <boost/format.hpp>

#include <boost/program_options.hpp>

#include <fenv.h>

#include <random>

#include "FArrayBox.H"
#include "VectorizedFABOps.H"
#include "HighResolutionTimer.H"

template <typename F>
void visit(FArrayBox& fab, F f)
{
    IntVect lower = fab.smallEnd();
    IntVect upper = fab.bigEnd(); 

    for (auto k = lower[2]; k <= upper[2]; ++k)
        for (auto j = lower[1]; j <= upper[1]; ++j)
            for (auto i = lower[0]; i <= upper[0]; ++i)
                f(fab(IntVect(i, j, k)), IntVect(i, j, k));
}

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    ///////////////////////////////////////////////////////////////////////////
    // Configuration

    std::uint64_t const nx   = vm["nx"].as<std::uint64_t>();
    std::uint64_t const it   = vm["it"].as<std::uint64_t>();
    std::uint64_t const seed = vm["seed"].as<std::uint64_t>();

    if (vm.count("header"))
        std::cout << "Extent (nx),Iterations (it),Seed,"
                     "Scalar Walltime [s],Vectorized Walltime [s]"
                  << std::endl;

    ///////////////////////////////////////////////////////////////////////////

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Box b(IntVect(0, 0, 0), IntVect(32, 32, nx));

    FArrayBox A(b, 1);
    FArrayBox B(b, 1);

    visit(A, [&](Real& val, IntVect) { val = dis(gen); });    
    visit(B, [&](Real& val, IntVect) { val = dis(gen); });    

    // Warmup.
    for (std::uint64_t i = 0; i < 256; ++i)
        A.plus(B, 17.0);

    HighResolutionTimer clock;

    for (std::uint64_t i = 0; i < it; ++i)
        A.plus(B, 17.0);

    Real scalar_walltime = clock.elapsed();

    // Warmup.
    for (std::uint64_t i = 0; i < 256; ++i)
        vectorizedPlus(B, A, 17.0);

    clock.restart();

    for (std::uint64_t i = 0; i < it; ++i)
        vectorizedPlus(B, A, 17.0);

    Real vectorized_walltime = clock.elapsed();

    std::cout << nx << ","
              << it << ","
              << seed << ","
              << scalar_walltime << ","
              << vectorized_walltime 
              << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    boost::program_options::options_description cmdline(
        "Vectorized FAB Operations Benchmark"
        );

    cmdline.add_options()
        ( "nx"
        , boost::program_options::value<std::uint64_t>()->
            default_value(32)
        , "extent of the box in the x direction - y and z are fixed as 32")
        ( "it"
        , boost::program_options::value<std::uint64_t>()->
            default_value(256)
        , "number of iterations to perform")
        ( "seed"
        , boost::program_options::value<std::uint64_t>()->
            default_value(1337)
        , "seed for pseudo random number generator")
        ( "header", "print header for the CSV timing data")
        ( "help", "print this information")
        ; 

    boost::program_options::variables_map vm;
    
    boost::program_options::store(
        boost::program_options::command_line_parser
            (argc, argv).options(cmdline).run(), vm);
    
    boost::program_options::notify(vm);
    
    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        std::exit(0);
    }

    return chombo_main(vm);
}

