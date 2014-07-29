/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012-2014 Bryce Adelstein-Lelbach aka wash 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_06FC5144_B4C1_4E16_9EC4_E00DD2FEDFB1)
#define CHOMBO_06FC5144_B4C1_4E16_9EC4_E00DD2FEDFB1

// init is the bootstrap procedure for launching HPX+Chombo applications.

#include <boost/program_options.hpp>

#include "NamespaceHeader.H"

typedef HPX_STD_FUNCTION<int(boost::program_options::variables_map&)>
    main_type;

// FIXME: Install an actual HPX log channel instead of hijacking std::cout

int runtime_main(
    main_type const& f
  , boost::program_options::variables_map& vm
    );

/// \brief Bootstrap Chombo's HPX services on all supported localities, and
/// invoke supplied \a f as application main.
///
/// \note This function will not return on UNIX platforms.
///
/// Remote-Parallel Operations:   Yes 
/// Local-Parallel Operations:    Yes 
/// Concurrency Control:          Futures 
/// Thread-Safety:                Thread-safe if f() is thread-safe 
/// Synchrony Gurantee:           Synchronous 
int init(
    main_type const& f,
    boost::program_options::options_description cmdline,
    int argc, char** argv,
    std::vector<std::string> const& cfg = std::vector<std::string>()
    )
#if !defined(BOOST_MSVC) && defined(__GNUC__)
    __attribute__ ((noreturn))
#endif
    ;

/// \brief Bootstrap the Chombo's HPX services on all supported localities, and
/// invoke supplied \a f as application main.
///
/// \note This function will not return on UNIX platforms.
///
/// Remote-Parallel Operations:   Yes 
/// Local-Parallel Operations:    Yes 
/// Concurrency Control:          Futures 
/// Thread-Safety:                Thread-safe if f() is thread-safe 
/// Synchrony Gurantee:           Synchronous 
inline int init(
    main_type const& f,
    int argc, char** argv,
    std::vector<std::string> const& cfg = std::vector<std::string>()
    )
{
    boost::program_options::options_description cmdline("HPX+Chombo AMR Driver");
    return init(f, cmdline, argc, argv, cfg); 
}

#include "NamespaceFooter.H"

#endif // CHOMBO_06FC5144_B4C1_4E16_9EC4_E00DD2FEDFB1

