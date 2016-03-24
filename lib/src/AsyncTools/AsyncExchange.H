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

#include "BoostSerialization.H"

#include "AsyncCopier.H"
#include "AsyncLevelData.H"
#include "AsyncLevelDataRegistry.H"
#include "DisjointBoxLayout.H"
#include "Ranges.H"

#include "NamespaceHeader.H"

// FIXME: Given that the other communications get their own class (e.g. interp,
// averaging), perhaps exchange should be removed from AsyncLevelData and these
// functions should be made methods of a new AsyncExchange class. The downside
// of this is that then we'd have to pass this class around everywhere.

// FIXME FIXME: Yes, turn this into a class.

// FIXME: Add communicate operator to AsyncLevelData.

// FIXME: We need to actually buffer the copy in the future or overlapping will
// break.

// FIXME: Is (from, to) the correct calling convention? memcpy is (to, from),
// but std::copy is (from, to).

// FIXME: LDOperator is virtual and awful.

// FIXME: Change comps to be Intervals.
template <typename T>
struct StoreRegion
{
  private:
    std::size_t epoch;
    std::size_t ct;
    int index_into_boxes;
    int index_into_data; 
    Interval to_comps;
    Box to_region;
    typename DataflowBox<T>::buffer data;
 
  public:
    StoreRegion()
      : epoch(0), ct(0), index_into_boxes(0), index_into_data(0)
      , to_comps(), to_region(), data()
    {}

    StoreRegion(
        std::size_t epoch_
      , std::size_t ct_
      , AsyncSendInstructions asi_
      , Interval to_comps_
      , Box to_region_
      , typename DataflowBox<T>::buffer&& data_
        )
      : epoch(epoch_), ct(ct_)
      , index_into_boxes(asi_.toIndex.intCode())
      , index_into_data(asi_.toIndex.datInd())
      , to_comps(to_comps_), to_region(to_region_)
      , data(data_)
    {}

    StoreRegion(StoreRegion const& rhs)
      : epoch(rhs.epoch), ct(rhs.ct)
      , index_into_boxes(rhs.index_into_boxes)
      , index_into_data(rhs.index_into_data)
      , to_comps(rhs.to_comps), to_region(rhs.to_region)
      , data(rhs.data)
    {}

    StoreRegion(StoreRegion&& rhs)
      : epoch(rhs.epoch), ct(rhs.ct)
      , index_into_boxes(rhs.index_into_boxes)
      , index_into_data(rhs.index_into_data)
      , to_comps(rhs.to_comps), to_region(rhs.to_region)
      , data(std::move(rhs.data))
    {}

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & epoch;
        ar & ct;
        ar & index_into_boxes;
        ar & index_into_data;
        ar & to_comps;
        ar & to_region;
        ar & data;
    }

    void operator()(AsyncLevelData<T>& ald) 
    {
        // Rebuild index.
        DataIndex const di = ald.disjointBoxLayout().localizeDataIndex(
            index_into_boxes, index_into_data
        ); 

        ald[di].store(epoch, ct, to_comps, to_region, std::move(data));
    }
};

template <typename T>
hpx::future<void> LocalPack(
    std::size_t epoch
  , std::size_t ct 
  , AsyncSendInstructions asi 
  , AsyncLevelData<T> const& from_ld
  , Interval from_comps
  , AsyncLevelData<T>& to_ld
  , Interval to_comps
    ) 
{
//    CH_assert(to_ld.nComp() == from_ld.nComp());

    CH_assert(to_comps.size() == from_comps.size());

    CH_assert(to_comps.size() <= to_ld.nComp());
    CH_assert(from_comps.size() <= from_ld.nComp());

    LDOperator<T> op;

    auto from_size = op.size(from_ld[asi.fromIndex], asi.fromRegion, from_comps);
    auto to_size   = op.size(to_ld[asi.toIndex], asi.toRegion, to_comps);

    CH_assert(from_size == to_size);

    typename DataflowBox<T>::buffer data(from_size);
    void* d = reinterpret_cast<void*>(data.data());

    op.linearOut(from_ld[asi.fromIndex], d, asi.fromRegion, from_comps);

//    to_ld[asi.toIndex].copy(
//        from_ld[asi.fromIndex]
//      , asi.fromRegion, /* starting comp */ from_comp
//      , asi.toRegion,   /* starting comp */ to_comp
//      , num_comp);

//    to_ld[asi.toIndex].store(epoch, ct, to_comps, asi.toRegion, std::move(data)); 

    unsigned const toProcID = to_ld.disjointBoxLayout().procID(asi.toIndex);

    if (procID() == toProcID)
    {
        to_ld[asi.toIndex].store(
            epoch, ct, to_comps, asi.toRegion, std::move(data)
        ); 

        return hpx::lcos::make_ready_future();
    }

    else
    {
        StoreRegion<T>
            sr(epoch, ct, asi, to_comps, asi.toRegion, std::move(data));
        hpx::util::function<void(AsyncLevelData<T>&)> f(sr);

        return ApplyALDAsync(toProcID, to_ld.tag(), std::move(f)); 
    }
}

template <typename T>
void LocalUnpack(
    std::size_t ct
  , Interval to_comps
  , Box to_region
  , DataflowBox<T>& to_box
  , typename DataflowBox<T>::buffer&& buf
    )
{
    LDOperator<T> op;

    void* b = reinterpret_cast<void*>(buf.data());

    op.linearIn(to_box, b, to_region, to_comps);  
}

// FIXME: Cache ASI lookup.
template <typename T>
hpx::future<void> LocalSendCopyAsync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , Interval from_comps
  , AsyncLevelData<T>& to_ld
  , Interval to_comps
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
        unsigned const fromProcID = dbl_from.procID(asi.fromIndex);
        unsigned const toProcID   = dbl_to.procID(asi.toIndex);

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
        auto F = LocalPack<T>;

        outbound.push_back(
            hpx::async(
                F, epoch, ct, asi
              , std::ref(from_ld), from_comps
              , std::ref(to_ld),   to_comps
            )
        );
    }

    return hpx::lcos::when_all(outbound);
}

// Copies all components.
template <typename T>
hpx::future<void> LocalSendCopyAsync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , AsyncLevelData<T>& to_ld
  , AsyncCopier const& ac 
    )
{
    return LocalSendCopyAsync(
        epoch, ct, di
      , from_ld, Interval(0, from_ld.nComp() - 1)
      , to_ld,   Interval(0, to_ld.nComp() - 1) 
      , ac
    );
}

template <typename T>
hpx::future<void> LocalRecvCopyAsync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T>& to_ld
  , Interval to_comps
    )
{
    auto F = LocalUnpack<T>;
    return to_ld[di].enqueueAll(epoch, ct, to_comps, F);
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
hpx::future<void> LocalCopyAsync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , Interval from_comps
  , AsyncLevelData<T>& to_ld
  , Interval to_comps
  , AsyncCopier const& ac 
    )
{
    hpx::future<void> outbound
        = LocalSendCopyAsync(
              epoch, ct, di 
            , from_ld, from_comps
            , to_ld,   to_comps
            , ac
          );

    hpx::future<void> inbound
        = LocalRecvCopyAsync(epoch, ct, di, to_ld, to_comps);

    return hpx::lcos::when_all(outbound, inbound);
}

// Copies all components.
template <typename T>
hpx::future<void> LocalCopyAsync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , AsyncLevelData<T>& to_ld
  , AsyncCopier const& ac 
    )
{
    return LocalCopyAsync(
        epoch, ct, di
      , from_ld, Interval(0, from_ld.nComp() - 1) 
      , to_ld,   Interval(0, to_ld.nComp() - 1)
      , ac
    );
}

template <typename T>
void LocalCopySync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , Interval from_comps
  , AsyncLevelData<T>& to_ld
  , Interval to_comps
  , AsyncCopier const& ac 
    )
{
    LocalCopyAsync(    
        epoch, ct, di 
      , from_ld, from_comps
      , to_ld,   to_comps
      , ac
    ).get();
}

// Copies all components.
template <typename T>
void LocalCopySync(
    std::size_t epoch
  , std::size_t ct 
  , DataIndex di 
  , AsyncLevelData<T> const& from_ld
  , AsyncLevelData<T>& to_ld
  , AsyncCopier const& ac 
    )
{
    LocalCopyAsync(epoch, ct, di, from_ld, to_ld, ac).get();
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
hpx::future<void> LocalExchangeAsync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
  , Interval from_comps
  , Interval to_comps
    )
{
    return LocalCopyAsync(
        epoch, Comm_ExchangeState, di
      , ld, from_comps
      , ld, to_comps
      , ld.exchangeCopier()
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
        epoch, Comm_ExchangeState, di, ld, ld, ld.exchangeCopier()
    );
}

template <typename T>
void LocalExchangeSync(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<T>& ld
  , Interval from_comps
  , Interval to_comps
    )
{
    LocalExchangeAsync(epoch, di, ld, from_comps, to_comps).get();
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
