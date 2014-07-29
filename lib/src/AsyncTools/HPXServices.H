/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012-2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_0A26F708_4EAE_4C4B_8D04_F88AF5D18F86)
#define CHOMBO_0A26F708_4EAE_4C4B_8D04_F88AF5D18F86

// Chombo-specific runtime services which augment the facilities exposed by the
// general-purpose HPX runtime system. These domain-specific extensions are
// written in place in large scientific codebases to suit their specific needs.
// Usually provide parallel I/O faciltities and global access to application
// configuration data taken from the command-line or input files.  
//
// Currently, this is very boring.

#include <vector>

#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/exception.hpp>

#include "NamespaceHeader.H"

struct HPXServicesServer;

// Only written to once per locality.
extern HPXServicesServer* runtime_ptr;

struct HPXServicesServer
  : hpx::components::simple_component_base<HPXServicesServer>
{
  private:
    std::vector<hpx::id_type> localities_;

  public:
    HPXServicesServer()
      : localities_()
    {
        HPX_THROW_IN_CURRENT_FUNC(hpx::not_implemented,
            "HPXServicesServer can't be default constructed");
    }

    HPXServicesServer(
        std::vector<hpx::id_type> const& localities
        );

    std::vector<hpx::id_type> const& localities() const
    {
        return localities_;
    }
};

///////////////////////////////////////////////////////////////////////////////
// API

/// \brief Retrieve a list of all localities that are running Chombo's HPX 
/// services. 
///
/// Remote-Parallel Operations:   None
/// Local-Parallel Operations:    None 
/// Concurrency Control:          Read-only access
/// Thread-Safety:                Guranteed if Chombo's HPX services is running 
/// Synchrony Gurantee:           Synchronous
inline std::vector<hpx::id_type> const& localities() 
{
    if (HPX_UNLIKELY(runtime_ptr == 0))
        HPX_THROW_IN_CURRENT_FUNC(hpx::invalid_status,
            "runtime_ptr is NULL");
    return runtime_ptr->localities(); 
}

/// \brief Invoke a function on all localities running the Chombo's HPX
/// services.
///
/// Remote-Parallel Operations:   Yes 
/// Local-Parallel Operations:    Yes 
/// Concurrency Control:          Futures 
/// Thread-Safety:                Thread-safe if f() is thread-safe 
/// Synchrony Gurantee:           Future-asynchronous
std::vector<hpx::future<void> > call_everywhere(
    hpx::util::function<void()> const& f
    );

#include "NamespaceFooter.H"

#endif // CHOMBO_0A26F708_4EAE_4C4B_8D04_F88AF5D18F86

