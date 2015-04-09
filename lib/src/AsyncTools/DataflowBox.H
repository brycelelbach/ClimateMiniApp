
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

enum CommType
{
    Comm_ExchangeState      = 0,
    Comm_CoarseToFineState  = 1,
    Comm_FineToCoarseState  = 2,
    Comm_FineToCoarseFlux   = 3,
    Comm_Last               = Comm_FineToCoarseFlux
};

template <typename T>
struct DataflowBox
{
    // These futures aren't actually storing any data, so we don't take a hit
    // by using shared futures instead of unique futures. This allows us to
    // set multi-region dependencies.
    typedef boost::shared_ptr<hpx::lcos::local::receive_buffer<void> > queue;

    // Regions that have external dependencies.
    std::array<std::map<Box, queue>, Comm_Last> regions; 
    
    T data;

    DataflowBox()
      : regions()
      , data()
    {} 

    DataflowBox(DataflowBox const& rhs) = delete;

    void addRegion(
        CommType ct
      , Box const& b
        )
    {
        queue q(new hpx::lcos::local::receive_buffer<void>());
        regions[ct].insert(std::make_pair(b, q));
    }

    template <typename F>
    hpx::future<void>& enqueue(
        CommType ct
      , std::size_t epoch
      , Box const& b
      , F&& f
        )
    {
        auto it = regions[ct].find(b);

        if (regions[ct].end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        auto WF = hpx::util::unwrapped(
            HPX_STD_BIND(std::forward<F>(f), HPX_STD_REF(data), b)
        );
        return (*it).second->receive(epoch).then(WF);
    }

    template <typename F>
    std::vector<hpx::future<void> > enqueueAll(
        CommType ct
      , std::size_t epoch
      , F&& f
        )
    {
        std::vector<hpx::future<void> > deps;
        deps.reserve(regions[ct].size());

        for (auto& kv : regions[ct])
        {
            auto WF = hpx::util::unwrapped(
                HPX_STD_BIND(std::forward<F>(f), HPX_STD_REF(data), kv.first)
            );
            deps.emplace_back(kv.second->receive(epoch).then(WF));
        }

        return deps; 
    }

    hpx::future<void> depend(
        CommType ct
      , std::size_t epoch
      , Box const& b
        )
    {
        auto it = regions[ct].find(b);

        if (regions[ct].end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        return (*it).second->receive(epoch);
    }

    std::vector<hpx::future<void> > dependAll(
        CommType ct
      , std::size_t epoch
        )
    {
        std::vector<hpx::future<void> > deps;
        deps.reserve(regions[ct].size());

        for (auto& kv : regions[ct])
            deps.push_back(kv.second->receive(epoch));

        return deps; 
    }

    void store(
        CommType ct
      , std::size_t epoch
      , Box const& b
        )
    {
        auto it = regions[ct].find(b);

        if (regions[ct].end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        (*it).second->store_received(epoch); 
    }
};

#include "NamespaceFooter.H"

#endif // CHOMBO_EBF8C34A_116B_4E81_866B_6C89121C1347
    
