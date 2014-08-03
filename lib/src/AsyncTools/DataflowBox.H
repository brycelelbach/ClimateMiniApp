
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

#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/local/receive_buffer.hpp>
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
    typedef std::shared_ptr<hpx::lcos::local::receive_buffer<void> > queue;

    // Regions that have external dependencies.
    std::map<Box, queue> regions; 
    
    T data;

    DataflowBox()
      : regions()
      , data()
    {} 

/*
    DataflowBox(DataflowBox const& rhs)
      : regions(rhs.regions)
      , data(rhs.data)
    {}
*/

    void addRegion(Box const& b)
    {
        queue q(new hpx::lcos::local::receive_buffer<void>());
        regions.insert(std::make_pair(b, q));
    }

    template <typename F>
    hpx::future<void>& enqueue(std::size_t step, Box const& b, F&& f)
    {
        auto it = regions.find(b);

        if (regions.end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        auto WF = hpx::util::unwrapped(
            HPX_STD_BIND(std::forward<F>(f), HPX_STD_REF(data), b)
        );
        return (*it).second->receive(step).then(WF);
    }

    template <typename F>
    std::vector<hpx::future<void> > enqueueAll(std::size_t step, F&& f)
    {
        std::vector<hpx::future<void> > deps;
        deps.reserve(regions.size());

        for (auto& kv : regions)
        {
            auto WF = hpx::util::unwrapped(
                HPX_STD_BIND(std::forward<F>(f), HPX_STD_REF(data), kv.first)
            );
            deps.emplace_back(kv.second->receive(step).then(WF));
        }

        return deps; 
    }

    hpx::future<void> depend(std::size_t step, Box const& b)
    {
        auto it = regions.find(b);

        if (regions.end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        return (*it).second->receive(step);
    }

    std::vector<hpx::future<void> > dependAll(std::size_t step)
    {
        std::vector<hpx::future<void> > deps;
        deps.reserve(regions.size());

        for (auto& kv : regions)
            deps.push_back(kv.second->receive(step));

        return deps; 
    }

    void store(std::size_t step, Box const& b)
    {
        auto it = regions.find(b);

        if (regions.end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        (*it).second->store_received(step); 
    }
};

#include "NamespaceFooter.H"

#endif // CHOMBO_EBF8C34A_116B_4E81_866B_6C89121C1347
    
