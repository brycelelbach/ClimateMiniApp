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

#include "StreamBox.H"
#include "AsyncLevelData.H"
#include "AsyncExchange.H"
#include "HPXDriver.H"

#include <boost/format.hpp>

template <typename F>
void visit(AsyncLevelData<FArrayBox>& soln, F f, unsigned comp)
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
                     , soln.disjointBoxLayout().procID(dit()));
    }
}

// FIXME: Should be variadic, and live in a header.
void output(
    std::string const& format
  , Vector<std::shared_ptr<AsyncLevelData<FArrayBox> > >& data
  , Vector<std::string> const& names
  , std::uint64_t step
  , Vector<int> const& ref_ratios
    )
{
    assert(data.size() != 0);
    assert(data[0].nComp() == names.size());

    std::uint64_t const levels = data.size();

    std::string file;
    try { file = boost::str(boost::format(format) % step); }
    catch (boost::io::too_many_args&) { file = format; }

    Vector<DisjointBoxLayout> dbl;
    Vector<LevelData<FArrayBox>*> ld;

    for (std::uint64_t L = 0; L < levels; ++L)
    {
        dbl.push_back(data[L]->disjointBoxLayout());

        ld.push_back(new LevelData<FArrayBox>);
        ld[L]->define(dbl[L], data[L]->nComp(), data[L]->ghostVect());

        DataIterator dit = data[L]->dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            auto& from = (*data[L])[dit()];
            auto& to   = (*ld[L])[dit()];

            // Alias from.
            to.define(Interval(0, data[L]->nComp()), from);
        }
    }

    WriteAMRHierarchyHDF5(
        file 
      , dbl 
      , ld
      , names
      , dbl[0].physDomain().domainBox() 
      , 1.0 // dx
      , 1.0 // dt
      , 0.0 // time
      , ref_ratios
      , levels 
        );

    for (LevelData<FArrayBox>* level : ld) delete level;
}

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    std::uint64_t constexpr procs = 1;
    std::uint64_t constexpr levels = 3;
    std::uint64_t constexpr dim = 16;
    std::uint64_t constexpr maxboxsize = 4;

    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    IntVect lower_bound(IntVect::Zero);
    IntVect upper_bound(
        (dim-1),
        (dim-1),
        (maxboxsize-1)
    );

    Vector<int> ref_ratios;
    for (std::uint64_t L = 0; L < levels; ++L)
        ref_ratios.push_back(2);

    bool is_periodic[] = { true, true, true };

    Vector<ProblemDomain> domain(levels);
    domain[0] = ProblemDomain(lower_bound, upper_bound, is_periodic);

    for (std::uint64_t L = 1; L < levels; ++L)
        domain[L] = refine(domain[L-1], ref_ratios[L]);

    Vector<Vector<Box> > box_list(levels);
    domainSplit(domain[0], box_list[0], maxboxsize, 1);

    Vector<IntVectSet> tags;
    Box btag = domain[0].domainBox();

    for (std::uint64_t L = 0; L < levels; ++L)
    {
        if (L != 0)
            btag = refine(btag, ref_ratios[L]); 

        IntVect shrink;
        shrink[0] = -((btag.bigEnd()[0] - btag.smallEnd()[0])+1)/4;
        shrink[1] = -((btag.bigEnd()[1] - btag.smallEnd()[0])+1)/4;
        shrink[2] = 0;

        btag.grow(shrink);

        tags.push_back(IntVectSet(btag));
    }

    std::cout << "\n";

    BRMeshRefine mesh_refine(
        domain[0]
      , ref_ratios
      , 1.0 // fill ratio
      , 1   // block factor
      , 1   // buffer size
      , maxboxsize 
        );

    Vector<Vector<Box> > new_box_list;
    mesh_refine.regrid(new_box_list, tags, 0, levels, box_list);
    box_list = new_box_list;

    assert(box_list.size() == levels);

    for (std::uint64_t L = 0; L < levels; ++L)
        std::cout << "L" << L << " BOX COUNT: " << box_list[L].size() << "\n";

    std::cout << "\n";

    Vector<Vector<int> > proc_list(levels);
    Vector<Vector<long> > load_list(levels);

    for (std::uint64_t L = 0; L < levels; ++L)
    {
        for (std::uint64_t j = 0; j < box_list[L].size(); ++j)
        {
            proc_list[L].push_back(0);
            load_list[L].push_back(box_list[L][j].volume());
        }

        mortonOrdering(box_list[L]);

        LoadBalance(proc_list[L], box_list[L], procs);
    }

    for (std::uint64_t L = 0; L < levels; ++L)
    {
        std::cout << "L" << L << " BOXES:\n";

        for (std::uint64_t j = 0; j < box_list[L].size(); ++j)
            std::cout << "  L" << L << " "
                      << streamBox(box_list[L][j])
                      << "\n";

        std::cout << "\n";
    }

    Vector<DisjointBoxLayout> dbl(levels);
    Vector<std::shared_ptr<AsyncLevelData<FArrayBox> > > data(levels);

    std::uint64_t boxCount = 0;
    std::uint64_t regionCount = 0;

    for (std::uint64_t L = 0; L < levels; ++L)
    {
        dbl[L].define(box_list[L], proc_list[L], domain[L], true);

        data[L].reset(new AsyncLevelData<FArrayBox>(dbl[L], 3, IntVect::Unit));

        // Components:
        // 0 - Box id
        // 1 - Received box ids
        // 2 - Region id

        DataIterator dit = data[L]->dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            // Give each box a unique id. 
            (*data[L])[dit()].setVal(
                ++boxCount, 0 
            );
        }

        auto neg1 = 
            [](Real& val, IntVect coords, unsigned procID)
            { val = -1.0; };

        visit(*data[L], neg1, 1);
        visit(*data[L], neg1, 2);

        for (AsyncRegion const& ar : data[L]->exchangeCopier().regions())
        {
            Box toBox = dbl[L][ar.toIndex];
    
            std::cout << "REGION: "
                      << streamBox(toBox)
                      << " CONTAINS " << streamBox(ar.toRegion) 
                      << "\n";
        
            // Give each region a unique id. 
            (*data[L])[ar.toIndex].setVal(
                ++regionCount, ar.toRegion, 2, 1
            );
        }
    
        std::cout << "\n";
    }

    for (std::uint64_t L = 0; L < levels; ++L)
    {
        std::vector<hpx::future<void> > local_exchanges;

        DataIterator dit = data[L]->dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            local_exchanges.push_back(
                LocalExchangeAsync(0, dit(), *data[L], 0, 1, 1)
            ); 
        }

        hpx::lcos::when_all(local_exchanges).get();
    }

    Vector<std::string> names;
    names.push_back("boxID");
    names.push_back("recvBoxID");
    names.push_back("regionID");

    output("mesh.%06u.hdf5", data, names, 0, ref_ratios);

/*
//
    DisjointBoxLayout dbl;
    dbl.define(box_list, proc_list, domain, true);

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
*/ 
    return 0;
}

int main(int argc, char** argv)
{
    return init(chombo_main, argc, argv); // Doesn't return
}

