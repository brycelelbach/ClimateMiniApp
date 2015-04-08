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

#include "AsyncLevelData.H"

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

    Box btag = base_domain.domainBox();
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

    std::cout << "EFFICIENCY: " << eff << "\n";

    std::cout << "PROC ASSIGNMENT:\n";

    for (unsigned i = 0; i < box_list.size(); ++i)
    {
        std::cout << " L" << i << "\n";
        for (unsigned j = 0; j < box_list[i].size(); ++j)
            std::cout << "  " << streamBox(box_list[i][j], proc_list[i][j])
                      << "\n";
    }

    std::cout << "\n";

    Vector<Vector<DisjointBoxLayout     > > dbl(procs);
    Vector<Vector<LayoutData<FArrayBox>*> > data(procs);

    for (size_t pid = 0; pid < procs; ++pid)
    {
        dbl[pid] = Vector<DisjointBoxLayout>(2);
        data[pid] = Vector<LayoutData<FArrayBox>*>(2);

        dbl[pid][0].define(box_list[0], proc_list[0], base_domain, true, pid);
        dbl[pid][1].define(box_list[1], proc_list[1], refine(base_domain, ref_ratios[1]), true, pid);

        data[pid][0] = new LayoutData<FArrayBox>();
        data[pid][1] = new LayoutData<FArrayBox>();
        data[pid][0]->define(dbl[pid][0]);
        data[pid][1]->define(dbl[pid][1]);

        DefineData(*data[pid][0], 3, IntVect::Unit);
        DefineData(*data[pid][1], 3, IntVect::Unit);

        auto setProcID =
            [](Real& val, IntVect coords, unsigned procID) { val = procID + 1; };

        visit(*data[pid][0], setProcID, 0);
        visit(*data[pid][1], setProcID, 0);

        auto neg1 = 
            [](Real& val, IntVect coords, unsigned procID) { val = -1.0; };

        visit(*data[pid][0], neg1, 1);
        visit(*data[pid][1], neg1, 1);
        visit(*data[pid][0], neg1, 2);
        visit(*data[pid][1], neg1, 2);
    }

    size_t regionCount = 0;

    for (size_t pid = 0; pid < procs; ++pid)
    {
        AsyncCopier ac;

        ac.exchangeDefine(dbl[pid][0], IntVect::Unit, pid);

        for (AsyncSendInstructions const& asi : ac.senderMotionPlan())
        {
            unsigned fromProcID = dbl[pid][0].procID(asi.fromIndex);
            unsigned toProcID   = dbl[pid][0].procID(asi.toIndex);

            DataIndex fromIdx
                = dbl[fromProcID][0].localizeDataIndex(asi.fromIndex); 
            DataIndex toIdx  
                = dbl[toProcID][0].localizeDataIndex(asi.toIndex); 

            Box fromBox = dbl[fromProcID][0][fromIdx];
            Box toBox   = dbl[toProcID][0][toIdx];

            std::cout << "SEND: "
                      << streamBox(fromBox, fromProcID)
                      << " " << streamBox(asi.fromRegion) 
                      << " -> "
                      << streamBox(toBox, toProcID)
                      << " " << streamBox(asi.toRegion) 
                      << std::endl;

            // Copy the processor ID to the test variable
            (*data[toProcID][0])[toIdx].copy(
                                          (*data[fromProcID][0])[fromIdx]
                                        , asi.fromRegion, 0
                                        , asi.toRegion, 1
                                        , 1);
        }

        std::cout << "\n";

        for (AsyncRegion const& ar : ac.regions())
        {
            unsigned toProcID = dbl[pid][0].procID(ar.toIndex);

            DataIndex toIdx = dbl[toProcID][0].localizeDataIndex(ar.toIndex); 

            Box toBox = dbl[toProcID][0][toIdx];

            std::cout << "REGION: "
                      << streamBox(toBox, toProcID)
                      << " " << streamBox(ar.toRegion) 
                      << std::endl;
    
            // Give each region a unique id. 
            (*data[toProcID][0])[toIdx].setVal(
                ++regionCount, ar.toRegion, 2, 1
            );
        }

        std::cout << "\n";
    }

    Vector<LevelData<FArrayBox>*> master_data(2);

    master_data[0] = new LevelData<FArrayBox>();
    master_data[1] = new LevelData<FArrayBox>();

    Vector<std::string> names;
    names.push_back("processor");
    names.push_back("sendInstructions");
    names.push_back("regions");

    for (size_t pid = 0; pid < procs; ++pid)
    {
        Vector<LevelData<FArrayBox>*> local_data(2);

        local_data[0] = new LevelData<FArrayBox>();
        local_data[1] = new LevelData<FArrayBox>();
        local_data[0]->define(dbl[pid][0], names.size(), IntVect::Unit);
        local_data[1]->define(dbl[pid][1], names.size(), IntVect::Unit);

        for (size_t level = 0; level < 2; ++level)
        {
            auto& src_level  = *data[pid][level];
            auto& dest_level = *local_data[level];

            DataIterator dit(src_level.dataIterator());
            for (dit.begin(); dit.ok(); ++dit)
            {
                auto& src  = src_level[dit()];
                auto& dest = dest_level[dit()];

                // Alias the src.
                dest.define(Interval(0, names.size()), src);
            }
        }

        WriteAMRHierarchyHDF5(
            boost::str(boost::format("mesh_L%06u_S%06u.hdf5") % pid % 0.0)
          , dbl[pid]
          , local_data
          , names
          , base_domain.domainBox()
          , 1.0 // dx
          , 1.0 // dt
          , 0.0 // time
          , ref_ratios
          , 2 // levels
            );

        delete local_data[0];
        delete local_data[1];
    }

    for (size_t pid = 0; pid < procs; ++pid)
    {
        delete data[pid][0];
        delete data[pid][1];
    }
}

