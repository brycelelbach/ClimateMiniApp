/*
 *      _______              __ 
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09)
#define CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09

#include <hpx/lcos/when_all.hpp>

#include <cstdint>

#include <assert.h>

#include "REAL.H"
#include "IntVect.H"
#include "FArrayBox.H"

#include "AsyncLevelData.H"
#include "AsyncExchange.H"

namespace climate_mini_app
{

struct problem_state
{
    problem_state()
    // {{{
      : U()
      , FY()
      , FZ()
      , epoch_()
    {} // }}}

    problem_state(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
    // {{{
      : U()
      , FY()
      , FZ()
      , epoch_()
    { 
        define(dbl, comps, ghost);
    } // }}}

    void define(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
    { // {{{
        U.define(dbl, comps, ghost);

        auto defineFY =
            [] (AsyncLevelData<FArrayBox>& ld, DataIndex di)
            {
                Box b = ld.disjointBoxLayout()[di];
                b.grow(ld.ghostVect());
                b = surroundingNodes(b, 1);
                ld[di].define(b, ld.nComp());
            }; 

        auto defineFZ =
            [] (AsyncLevelData<FArrayBox>& ld, DataIndex di)
            {
                Box b = ld.disjointBoxLayout()[di];
                b.grow(ld.ghostVect());
                b = surroundingNodes(b, 2);
                ld[di].define(b, ld.nComp());
            }; 
 
        FY.define(dbl, comps, ghost, defineFY);
        FZ.define(dbl, comps, ghost, defineFZ);

        epoch_.define(dbl);

        DataIterator dit(U.dataIterator());

        for (dit.begin(); dit.ok(); ++dit)
        {
            epoch_[dit()] = 0;
        }
    } // }}}

    hpx::future<void> exchangeAllAsync()
    { // {{{
        DataIterator dit = U.dataIterator();

        std::vector<hpx::future<void> > exchanges;
    
        for (dit.begin(); dit.ok(); ++dit)
            exchanges.push_back(LocalExchangeAsync(epoch_[dit()]++, dit(), U));

        return hpx::lcos::when_all(exchanges);
    } // }}}

    void exchangeAllSync()
    { // {{{
        exchangeAllAsync().get();
    } // }}}

    AsyncLevelData<FArrayBox> U;
    AsyncLevelData<FArrayBox> FY;
    AsyncLevelData<FArrayBox> FZ;

  private:
    LayoutData<std::size_t> epoch_;

  public:
    friend struct sub_problem_state;
};

struct sub_problem_state
{
    sub_problem_state()
    // {{{
      : ps_(0)
      , di_()
    {} // }}}

    sub_problem_state(problem_state& ps, DataIndex di)
    // {{{
      : ps_(&ps)
      , di_(di)
    {} // }}}

    sub_problem_state(sub_problem_state const& other)
    // {{{
      : ps_(other.ps_)
      , di_(other.di_)
    {} // }}}

    sub_problem_state& operator=(sub_problem_state const& other)
    { // {{{
        ps_ = other.ps_;
        di_ = other.di_;
        return *this;
    } // }}}

    void copy(sub_problem_state const& A)
    { // {{{
        U().copy(A.U());
        FY().copy(A.FY());
        FZ().copy(A.FZ());
    } // }}}

    void setVal(Real val)
    { // {{{
        U().setVal(val);
        FY().setVal(val);
        FZ().setVal(val);
    } // }}}

    void plus(sub_problem_state const& A, Real factor = 1.0)
    { // {{{
        U().plus(A.U(), factor);
    } // }}}

    hpx::future<void> exchangeAsync()
    { // {{{
        assert(ps_);
        return LocalExchangeAsync(ps_->epoch_[di_]++, di_, ps_->U);
    } // }}}

    void exchangeSync()
    { // {{{
        assert(ps_);
        LocalExchangeSync(ps_->epoch_[di_]++, di_, ps_->U);
    } // }}}

    FArrayBox& U()
    { // {{{
        assert(ps_);
        return ps_->U[di_]; 
    } // }}}

    FArrayBox const& U() const
    { // {{{
        assert(ps_);
        return ps_->U[di_]; 
    } // }}}

    FArrayBox& FY()
    { // {{{
        assert(ps_);
        return ps_->FY[di_]; 
    } // }}}

    FArrayBox const& FY() const
    { // {{{
        assert(ps_);
        return ps_->FY[di_]; 
    } // }}}

    FArrayBox& FZ()
    { // {{{
        assert(ps_);
        return ps_->FZ[di_]; 
    } // }}}

    FArrayBox const& FZ() const
    { // {{{
        assert(ps_);
        return ps_->FZ[di_]; 
    } // }}}

  private:
    problem_state* ps_;
    DataIndex di_;
};

}

#endif // CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09

