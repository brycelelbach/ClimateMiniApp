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

#if !defined(CHOMBO_BFB0AB25_CB31_41FD_8990_164847DC199E)
#define CHOMBO_BFB0AB25_CB31_41FD_8990_164847DC199E

#include <hpx/async.hpp>
#include <hpx/lcos/when_all.hpp>

#include "AsyncCopier.H"
#include "AsyncLevelData.H"
#include "DisjointBoxLayout.H"
#include "Ranges.H"

#include "NamespaceHeader.H"

template <typename T>
void LocalCopyAsync(
    std::size_t epoch
  , AsyncSendInstructions asi 
  , AsyncLevelData<T>& ld
    ) 
{
    CH_assert(ld[asi.toIndex].data.nComp() == ld[asi.fromIndex].data.nComp());

    ld[asi.toIndex].data.copy(
        ld[asi.fromIndex].data
      , asi.fromRegion, /* starting comp */ 0
      , asi.toRegion,   /* starting comp */ 0
      , ld[asi.fromIndex].data.nComp());

    ld[asi.toIndex].store(epoch, asi.toRegion); 
}

// FIXME: Cache ASI lookup.
template <typename T>
std::vector<hpx::future<void> > SendLocalExchangeAsync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
    )
{
    DisjointBoxLayout const& dbl = ld.disjointBoxLayout(); 
    AsyncCopier const& ac = ld.exchangeCopier();
    auto const& smp = ac.senderMotionPlan();

    std::vector<hpx::future<void> > outbound;

    AsyncSendInstructions key(di);
    for (AsyncSendInstructions const& asi : asRange(smp.equal_range(key))) 
    {
        unsigned fromProcID = dbl.procID(asi.fromIndex);
        unsigned toProcID   = dbl.procID(asi.toIndex);

        CH_assert(toProcID == fromProcID);

        auto F = LocalCopyAsync<T>;

        outbound.push_back(
            hpx::async(F, epoch, asi, std::ref(ld))
        );
    }

    return outbound;
}

template <typename T>
std::vector<hpx::future<void> > RecvLocalExchangeAsync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
    )
{
    return ld[di].dependAll(epoch);
}

template <typename T>
void LocalExchangeSync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
    )
{
    std::vector<hpx::future<void> > outbound
        = SendLocalExchangeAsync(epoch, di, ld);
    std::vector<hpx::future<void> > inbound
        = RecvLocalExchangeAsync(epoch, di, ld);

    hpx::lcos::when_all(outbound).get();
    hpx::lcos::when_all(inbound).get();
}


#include "NamespaceFooter.H"

#endif // CHOMBO_BFB0AB25_CB31_41FD_8990_164847DC199E
