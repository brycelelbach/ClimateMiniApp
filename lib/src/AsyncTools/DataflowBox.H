
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
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/unwrapped.hpp>

#include <boost/array.hpp>

#include "Box.H"

#include "NamespaceHeader.H"

enum CommType
{
    Comm_Invalid            = -1,
    Comm_ExchangeState      = 0,
    Comm_CoarseToFineState  = 1,
    Comm_FineToCoarseState  = 2,
    Comm_FineToCoarseFlux   = 3
};

// FIXME: Remove shared_ptr
// FIXME: Nested maps?

template <typename T>
struct DataflowBox : T
{
    typedef std::tuple<
        /* CommType */ std::size_t, /* 'To' Components */ Interval, Box
    > key_type;

    typedef std::vector<std::uint8_t> buffer;

    typedef boost::shared_ptr<hpx::lcos::local::receive_buffer<buffer> > queue;

  private:
    // Regions that have external dependencies.
    std::map<key_type, queue> regions; 

  public:
    DataflowBox()
      : T()
      , regions()
    {} 

    DataflowBox(DataflowBox const& rhs) = delete;

    void addRegion(
        std::size_t ct
      , Interval comps
      , Box const& b
        )
    {
        queue q(new hpx::lcos::local::receive_buffer<buffer>());
        regions.insert(std::make_pair(key_type(ct, comps, b), q));
    }

    template <typename F>
    hpx::future<void>& enqueue(
        std::size_t epoch
      , std::size_t ct
      , Interval comps
      , Box const& b
      , F&& f
        )
    {
        auto it = regions.find(key_type(ct, comps, b));

        if (regions.end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        auto WF = hpx::util::unwrapped(std::bind(
            std::forward<F>(f), ct, comps, b
                              , std::ref(*this), std::placeholders::_1
        ));
        return (*it).second->receive(epoch).then(WF);
    }

    // FIXME: O(n)
    template <typename F>
    hpx::future<void> enqueueAll(
        std::size_t epoch
      , std::size_t ct
      , Interval comps
      , F&& f
        )
    {
        std::vector<hpx::future<void> > deps;
        deps.reserve(regions.size());

        for (auto& kv : regions)
        {
            if (  (std::get<0>(kv.first) == ct)
               && (std::get<1>(kv.first) == comps))
            {
                auto WF = hpx::util::unwrapped(std::bind(
                    std::forward<F>(f), ct, comps, std::get<2>(kv.first)
                                      , std::ref(*this), std::placeholders::_1
                ));
                deps.emplace_back(kv.second->receive(epoch).then(WF));
            }
        }

        return hpx::lcos::when_all(deps); 
    }

    hpx::future<buffer> depend(
        std::size_t epoch
      , std::size_t ct
      , Interval comps
      , Box const& b
        )
    {
        auto it = regions.find(key_type(ct, comps, b));

        if (regions.end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        return (*it).second->receive(epoch);
    }

    void store(
        std::size_t epoch
      , std::size_t ct
      , Interval comps
      , Box const& b
      , buffer&& buf
        )
    {
        auto it = regions.find(key_type(ct, comps, b));

        if (regions.end() == it)
            HPX_THROW_STD_EXCEPTION(
                std::out_of_range("no such region\n")
              , BOOST_CURRENT_FUNCTION);

        (*it).second->store_received(epoch, std::move(buf)); 
    }
};

#include "NamespaceFooter.H"

#endif // CHOMBO_EBF8C34A_116B_4E81_866B_6C89121C1347
    
