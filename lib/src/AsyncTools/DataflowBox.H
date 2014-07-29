
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

#if !defined(CHOMBO_EBF8C34A_116B_4E81_866B_6C89121C1347)
#define CHOMBO_EBF8C34A_116B_4E81_866B_6C89121C1347

#include <hpx/lcos/future.hpp>
#include <hpx/util/unwrapped.hpp>

#include "Box.H"

#include "NamespaceHeader.H"

template <typename T>
struct DataflowBox
{
    // These futures aren't actually storing any data, so we don't take a hit
    // by using shared futures instead of unique futures. This allows us to
    // set multi-region dependencies.
    typedef hpx::shared_future<void> queue;

    // Regions that have external dependencies.
    std::map<Box, queue> regions; 
    
    T data;

    DataflowBox()
      : regions()
      , data()
    {} 

    DataflowBox(DataflowBox const& rhs)
      : regions(rhs.regions)
      , data(rhs.data)
    {}

    void addRegion(Box const& b)
    {
        regions.insert(std::make_pair(b, hpx::make_ready_future()));
    }

    template <typename F>
    queue& enqueue(Box const& b, F&& f)
    {
        auto it = regions.find(b);

        if (regions.end() != it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        auto WF = hpx::util::unwrapped(
            HPX_STD_BIND(std::forward<F>(f), HPX_STD_REF(data), b)
        );
        return (*it).second = (*it).second.then(WF);
    }

    template <typename F>
    std::vector<queue> enqueueAll(F&& f)
    {
        std::vector<queue> deps;
        deps.reserve(regions.size());

        for (auto& kv : regions)
        {
            auto WF = hpx::util::unwrapped(
                HPX_STD_BIND(std::forward<F>(f), HPX_STD_REF(data), kv.first)
            );
            kv.second = kv.second.then(WF);
            deps.push_back(kv.second);
        }

        return deps; 
    }
};
    
#include "NamespaceFooter.H"

#endif // CHOMBO_EBF8C34A_116B_4E81_866B_6C89121C1347
    
