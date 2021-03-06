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

#include <mpi.h>

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/serialization/vector.hpp>

#include "HPXDriver.H"
#include "HPXServices.H"

#include "NamespaceHeader.H"

using boost::program_options::options_description;
using boost::program_options::variables_map;
using boost::program_options::value;

using hpx::components::stubs::runtime_support;

int runtime_main(main_type const& f, variables_map& vm)
{
    int result = 0;

    {
#if defined(DEBUG) 
        // For great justice (lack of Octopus ASCII art is saddening).
        std::cout << "Bootstrapping Chombo's HPX services\n"
                  << std::flush;
#endif
 
        ///////////////////////////////////////////////////////////////////////
        // Initialize Chombo's HPX services.

        // Lookup the server component type in AGAS
        hpx::components::component_type type =
            hpx::components::get_component_type<HPXServicesServer>();
    
        // Find all localities supporting the component 
        std::vector<hpx::id_type> localities = hpx::find_all_localities(type);

        if (localities.empty())
            HPX_THROW_IN_CURRENT_FUNC(hpx::assertion_failure,
                "no localities supporting Chombo's HPX services");
   
#if defined(DEBUG) 
        std::cout << "Found " << localities.size()
                  << " usable localities, deploying service components\n"
                  << std::flush;
#endif

        // Asynchronously deploy the component on every locality.
        // FIXME: Sadly, distributing factory doesn't support constructor args
        // yet, so we have to do this by hand.
        std::vector<hpx::future<hpx::id_type> > big_boot_barrier;    
        big_boot_barrier.reserve(localities.size());

        for (std::size_t i = 0; i < localities.size(); ++i)
        {
            // Create the component, passing any relevant global information.
            big_boot_barrier.emplace_back(
                runtime_support::create_component_async<HPXServicesServer>
                    (localities[i], localities));
        }

        // Wait on B^3.
        hpx::when_all(big_boot_barrier).get(); 

        ///////////////////////////////////////////////////////////////////////
        // Invoke user entry point

#if defined(DEBUG)
        std::cout << "Chombo services are live, executing application code\n"
                  << std::flush;
#endif

        result = f(vm);

#if defined(DEBUG)
        std::cout << "Application execution complete, initiating shutdown\n"
                  << std::flush;
#endif
    }

    hpx::finalize();
    return result;
}

int init(
    main_type const& f,
    boost::program_options::options_description cmdline,
    int argc, char** argv,
    std::vector<std::string> const& cfg
    )
{
#if defined(DEBUG)
    std::cout << "Launching HPX runtime\n" << std::flush;
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Initialize HPX.
    main_type Lf(std::bind(&runtime_main, f, std::placeholders::_1)); 
    int result = hpx::init(Lf, cmdline, argc, argv, cfg); 

#if defined(DEBUG)
    std::cout << "HPX runtime has shutdown\n" << std::flush;
#endif

    return result;
}

#include "NamespaceFooter.H"

