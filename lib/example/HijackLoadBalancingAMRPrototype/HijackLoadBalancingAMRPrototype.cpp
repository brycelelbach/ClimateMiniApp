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
#include "AMRIO.H"

struct streamBox
{
    Box const& b;

    streamBox(Box const& b_) : b(b_) {}

    friend std::ostream& operator<<(std::ostream& os, streamBox const& sb)
    {
        return os << "(" << sb.b.smallEnd() << " " << sb.b.bigEnd()
                  << " " << sb.b.volume() << ")";
    } 
};

template <typename F>
void visit(LevelData<FArrayBox>& soln, F f)
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
                    f( subsoln(IntVect(i, j, k))
                     , IntVect(i, j, k)
                     , soln.boxLayout().procID(dit()));
    }
}

int main()
{
    size_t constexpr dim = 2;
    size_t constexpr procs = 8;
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

    Vector<Vector<Box> > box_list(2);
    domainSplit(base_domain, box_list[0], maxboxsize, 1);

    Vector<int> ref_ratios;
    ref_ratios.push_back(2);
    ref_ratios.push_back(2);

    BRMeshRefine mesh_refine(
        base_domain
      , ref_ratios
      , 1.0 // fill ratio
      , 1   // block factor
      , 1   // buffer size
      , maxboxsize 
        );

    Box btag = base_domain.domainBox();//refine(base_domain.domainBox(), ref_ratios[1]);
    IntVect shrink;
    shrink[0] = -btag.bigEnd()[0]/4;
    shrink[1] = -btag.bigEnd()[1]/4;
    shrink[2] = 0;
    btag.grow(shrink);
    IntVectSet tags(btag);

    std::cout << "REFINE REGION: " << streamBox(btag) << "\n\n";

    Vector<Vector<Box> > new_box_list;
    mesh_refine.regrid(new_box_list, tags, 0, 0, box_list);
    box_list = new_box_list;

    std::cout << "L0 BOXES: " << box_list[0].size() << "\n"
              << "L1 BOXES: " << box_list[1].size() << "\n"
              << "\n";

    Vector<Vector<int> > proc_list(2);
    Vector<Vector<long> > load_list(2);

    for (unsigned i = 0; i < box_list.size(); ++i)
        for (unsigned j = 0; j < box_list[i].size(); ++j)
            proc_list[i].push_back(0);

    for (unsigned i = 0; i < box_list.size(); ++i)
        for (unsigned j = 0; j < box_list[i].size(); ++j)
            load_list[i].push_back(box_list[i][j].volume());

    Real eff = 0.0;

    mortonOrdering(box_list[0]);
    mortonOrdering(box_list[1]);

    LoadBalance(proc_list[0], box_list[0], procs);
    LoadBalance(proc_list[1], box_list[1], procs);
//    LoadBalance(proc_list, eff, box_list, load_list, ref_ratios, procs);

    std::cout << "EFFICIENCY: " << eff << "\n";

    std::cout << "PROC ASSIGNMENT:\n";

    for (unsigned i = 0; i < box_list.size(); ++i)
    {
        std::cout << " L" << i << "\n";
        for (unsigned j = 0; j < box_list[i].size(); ++j)
            std::cout << "  " << streamBox(box_list[i][j])
                      << " -> " << proc_list[i][j] << "\n";
    }

    std::cout << "\n";

    Vector<DisjointBoxLayout> dbl(2);
    dbl[0].define(box_list[0], proc_list[0], base_domain, true);
    dbl[1].define(box_list[1], proc_list[1], refine(base_domain, ref_ratios[1]), true);

    Vector<LevelData<FArrayBox>*> data(2);
    data[0] = new LevelData<FArrayBox>();
    data[1] = new LevelData<FArrayBox>();
    data[0]->define(dbl[0], 1, IntVect::Zero);
    data[1]->define(dbl[1], 1, IntVect::Zero);

    auto setProcID =
        [](Real& val, IntVect coords, unsigned procID) { val = procID; };

    visit(*data[0], setProcID);
    visit(*data[1], setProcID);

    Vector<std::string> names;
    names.push_back("processor");

    WriteAMRHierarchyHDF5(
        "mesh.hdf5"
      , dbl
      , data
      , names
      , base_domain.domainBox()
      , 1.0
      , 1.0
      , 0.0
      , ref_ratios
      , 2
        );

    delete data[0];
    delete data[1];
}

