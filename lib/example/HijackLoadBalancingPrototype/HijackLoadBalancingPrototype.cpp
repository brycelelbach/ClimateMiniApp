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

#include "IntVect.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"
#include "NeighborIterator.H"
#include "LevelData.H"
#include "FArrayBox.H"

int main()
{
    size_t constexpr dim = 8;
    size_t constexpr procs = 8;
    size_t constexpr maxboxsize = 4;

    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    IntVect lower_bound(IntVect::Zero);
    IntVect upper_bound(
        (dim*procs-1),
        (maxboxsize-1),
        (maxboxsize-1)
    );

    ProblemDomain base_domain(lower_bound, upper_bound);

    Vector<Box> box_list;
    domainSplit(base_domain, box_list, maxboxsize, 1);

    std::cout << "BOXES: " << box_list.size() << "\n\n";

    Vector<int> proc_list(box_list.size(), 0);

    LoadBalance(proc_list, box_list, procs);

    std::cout << "PROC ASSIGNMENT:\n";

    for (auto i = 0; i < box_list.size(); ++i) 
        std::cout << "  " << box_list[i] << " -> " << proc_list[i] << "\n";

    std::cout << "\n";

    DisjointBoxLayout dbl(box_list, proc_list, base_domain, true);

    NeighborIterator nit(dbl);
    DataIterator dit = dbl.dataIterator();

    for (dit.begin(); dit.ok(); ++dit)
    {
        auto const& bidx = dit();
        Box b = dbl[bidx]; 
        auto bpid = dbl.procID(bidx);

        std::cout << "BOX: " << b << " N" << bpid << "\n";

        for (nit.begin(bidx); nit.ok(); ++nit) 
        {
            auto const& nidx = nit();
            Box n = dbl[nidx]; 
            auto npid = dbl.procID(nidx);
            std::cout << "  NEIGHBOR: " << n << " N" << npid << "\n";
        }
    }
}

