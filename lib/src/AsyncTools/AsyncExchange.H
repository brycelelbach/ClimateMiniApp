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

// FIXME: Given that the other communications get their own class (e.g. interp,
// averaging), perhaps exchange should be removed from AsyncLevelData and these
// functions should be made methods of a new AsyncExchange class. The downside
// of this is that then we'd have to pass this class around everywhere.

// FIXME: Add communicate operator to AsyncLevelData.

template <typename T>
void LocalCopyImpl(
    CommType ct
  , std::size_t epoch
  , AsyncSendInstructions asi 
  , AsyncLevelData<T> const& from_ld
  , unsigned from_comp
  , AsyncLevelData<T>& to_ld
  , unsigned to_comp
  , unsigned num_comp
    ) 
{
    CH_assert(
        to_ld[asi.toIndex].nComp() == from_ld[asi.fromIndex].nComp()
    );

    to_ld[asi.toIndex].copy(
        from_ld[asi.fromIndex]
      , asi.fromRegion, /* starting comp */ from_comp
      , asi.toRegion,   /* starting comp */ to_comp
      , num_comp);

    to_ld[asi.toIndex].store(ct, epoch, asi.toRegion); 
}

// FIXME: Cache ASI lookup.
template <typename T>
hpx::future<void> LocalSendCopyAsync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , unsigned from_comp
  , AsyncLevelData<T>& to_ld
  , unsigned to_comp
  , unsigned num_comp
  , AsyncCopier const& ac 
    )
{
    DisjointBoxLayout const& dbl_from = from_ld.disjointBoxLayout(); 
    DisjointBoxLayout const& dbl_to   = to_ld.disjointBoxLayout(); 
    auto const& smp = ac.senderMotionPlan();

    std::vector<hpx::future<void> > outbound;

    AsyncSendInstructions key(di);
    for (AsyncSendInstructions const& asi : asRange(smp.equal_range(key))) 
    {
        unsigned fromProcID = dbl_from.procID(asi.fromIndex);
        unsigned toProcID   = dbl_to.procID(asi.toIndex);

        CH_assert(toProcID == fromProcID);

/*
        void (&F)( CommType
                 , std::size_t
                 , AsyncSendInstructions
                 , AsyncLevelData<T> const&
                 , unsigned
                 , AsyncLevelData<T>& 
                 , unsigned
                 , unsigned
                 )
*/
        auto F = LocalCopyImpl<T>;

        outbound.push_back(
            hpx::async(
                F, ct, epoch, asi
              , std::ref(from_ld), from_comp
              , std::ref(to_ld), to_comp
              , num_comp
            )
        );
    }

    return hpx::lcos::when_all(outbound);
}

// Copies all components.
template <typename T>
hpx::future<void> LocalSendCopyAsync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , AsyncLevelData<T>& to_ld
  , AsyncCopier const& ac 
    )
{
    unsigned const num_comp = from_ld.nComp();
    return LocalSendCopyAsync(ct, epoch, asi, from_ld, 0, to_ld, 0, num_comp);
}

template <typename T>
hpx::future<void> LocalRecvCopyAsync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& to_ld
    )
{
    return to_ld[di].dependAll(ct, epoch);
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
hpx::future<void> LocalCopyAsync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , unsigned from_comp
  , AsyncLevelData<T>& to_ld
  , unsigned to_comp
  , unsigned num_comp
  , AsyncCopier const& ac 
    )
{
    hpx::future<void> outbound
        = LocalSendCopyAsync(
              ct, epoch, di 
            , from_ld, from_comp
            , to_ld, to_comp
            , num_comp, ac
          );

    hpx::future<void> inbound
        = LocalRecvCopyAsync(ct, epoch, di, to_ld);

    return hpx::lcos::when_all(outbound, inbound);
}

// Copies all components.
template <typename T>
hpx::future<void> LocalCopyAsync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , AsyncLevelData<T>& to_ld
  , AsyncCopier const& ac 
    )
{
    unsigned const num_comp = from_ld.nComp();
    return LocalCopyAsync(ct, epoch, di, from_ld, 0, to_ld, 0, num_comp, ac);
}

template <typename T>
void LocalCopySync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , unsigned from_comp
  , AsyncLevelData<T>& to_ld
  , unsigned to_comp
  , unsigned num_comp
  , AsyncCopier const& ac 
    )
{
    LocalCopyAsync(    
        ct, epoch, di 
      , from_ld, from_comp
      , to_ld, to_comp
      , num_comp, ac
    ).get();
}

// Copies all components.
template <typename T>
void LocalCopySync(
    CommType ct
  , std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , AsyncLevelData<T>& to_ld
  , AsyncCopier const& ac 
    )
{
    LocalCopyAsync(ct, epoch, di, from_ld, to_ld, ac).get();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
hpx::future<void> LocalExchangeAsync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
  , unsigned from_comp
  , unsigned to_comp
  , unsigned num_comp
    )
{
    return LocalCopyAsync(
        Comm_ExchangeState, epoch, di
      , ld, from_comp
      , ld, to_comp
      , num_comp, ld.exchangeCopier()
    );
}

// Copies all components.
template <typename T>
hpx::future<void> LocalExchangeAsync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
    )
{
    return LocalCopyAsync(
        Comm_ExchangeState, epoch, di, ld, ld, ld.exchangeCopier()
    );
}

template <typename T>
void LocalExchangeSync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
  , unsigned from_comp
  , unsigned to_comp
  , unsigned num_comp
    )
{
    LocalExchangeAsync(epoch, di, ld, from_comp, to_comp, num_comp).get();
}

// Copies all components.
template <typename T>
void LocalExchangeSync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
    )
{
    LocalExchangeAsync(epoch, di, ld).get();
}

#include "NamespaceFooter.H"

#endif // CHOMBO_BFB0AB25_CB31_41FD_8990_164847DC199E
