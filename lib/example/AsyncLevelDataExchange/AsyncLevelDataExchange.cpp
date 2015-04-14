/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <fenv.h>

#include <hpx/lcos/wait_all.hpp>

#include "UsingBaseNamespace.H"

#include "IntVect.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"
#include "NeighborIterator.H"
#include "LevelData.H"
#include "FArrayBox.H"
#include "AMRIO.H"

#include "AsyncLevelData.H"
#include "AsyncExchange.H"
#include "HPXDriver.H"

#include <boost/format.hpp>

struct streamBox
{
    Box const& b;
    int proc;

    streamBox(Box const& b_, int proc_ = -1) : b(b_), proc(proc_) {}

    friend std::ostream& operator<<(std::ostream& os, streamBox const& sb)
    {
        if (-1 == sb.proc) 
            return os << "(" << sb.b.smallEnd() << " " << sb.b.bigEnd()
                      << " " << sb.b.volume() << ")";
        else
            return os << "(L" << sb.proc
                      << " " << sb.b.smallEnd() << " " << sb.b.bigEnd()
                      << " " << sb.b.volume() << ")";
    } 
};

template <typename F>
void visit(LayoutData<FArrayBox>& soln, F f, unsigned comp)
{
    DataIterator dit = soln.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    { 
        auto& subsoln = soln[dit];
        IntVect lower = subsoln.smallEnd();
        IntVect upper = subsoln.bigEnd(); 

        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    f( subsoln(IntVect(i, j, k), comp)
                     , IntVect(i, j, k)
                     , soln.boxLayout().procID(dit()));
    }
}

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    size_t constexpr dim = 16;
    size_t constexpr procs = 1;
    size_t constexpr maxboxsize = 4;

    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    IntVect lower_bound(IntVect::Zero);
    IntVect upper_bound(
        (dim*procs-1),
        (dim*procs-1),
        (maxboxsize-1)
    );

    ProblemDomain base_domain(lower_bound, upper_bound);

    Vector<Box> box_list;
    domainSplit(base_domain, box_list, maxboxsize, 1);

    Vector<int> proc_list;

    for (unsigned i = 0; i < box_list.size(); ++i)
        proc_list.push_back(0);

    mortonOrdering(box_list);

    LoadBalance(proc_list, box_list, procs);

    std::cout << "BOXES:\n";

    for (unsigned i = 0; i < box_list.size(); ++i)
        std::cout << " " << streamBox(box_list[i], proc_list[i]) << "\n";

    std::cout << "\n";

    DisjointBoxLayout dbl;
    dbl.define(box_list, proc_list, base_domain, true);

    AsyncLevelData<FArrayBox> ald(dbl, IntVect::Unit);
    DefineData(ald, 1);

    std::vector<hpx::future<void> > exchanges;

    DataIterator dit = ald.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        exchanges.push_back(
            hpx::async(LocalExchangeSync<FArrayBox>, 0, dit(), std::ref(ald))
        ); 
    }

    for (hpx::future<void>& f : exchanges) f.get();
 
    return 0;
}

int main(int argc, char** argv)
{
    return init(chombo_main, argc, argv); // Doesn't return
}

