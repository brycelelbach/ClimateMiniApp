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

#include "heat3d.hpp"
#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"
#include "ARK4.H"

#include <fenv.h>

#include <boost/format.hpp>

template <typename F>
void visit(heat3d::problem_state& soln, F f)
{
    DataIterator dit = soln.data().dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    { 
        auto& subsoln = soln.data()[dit];
        IntVect lower = subsoln.smallEnd();
        IntVect upper = subsoln.bigEnd(); 

        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    f(subsoln(IntVect(i, j, k)), IntVect(i, j, k));
    }
}

void output(heat3d::problem_state const& soln, std::string const& format, std::uint64_t step)
{
    std::string file = boost::str(boost::format(format) % step);
    writeLevelname(&soln.data(), file.c_str());
}

void output(FArrayBox const& soln, std::string const& format, std::uint64_t step)
{
    std::string file = boost::str(boost::format(format) % step);
    writeFABname(&soln, file.c_str(), Vector<std::string>(1, "phi"));
}

int main()
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    heat3d::configuration config(
        /*nt: physical time to step to             =*/0.003,
        /*nh: y and z (horizontal) extent per core =*/40,
        /*nv: x (vertical) extent per core         =*/2,
        /*max_box_size                             =*/40
    );

    heat3d::aniso_profile profile(config,
        // sine factors in the source term
        /*A=*/1.0,   /*B=*/1.0,   /*C=*/1.0,

        // diffusion coefficients
        /*kx=*/0.5, /*ky=*/0.75, /*kz=*/0.75
    ); 

    IntVect lower_bound(IntVect::Zero);
    IntVect upper_bound(
        (config.nv*numProc()-1),
        (config.nh*numProc()-1),
        (config.nh*numProc()-1)
    );

    ProblemDomain base_domain(lower_bound, upper_bound);

    Vector<Box> boxes;
    domainSplit(base_domain, boxes, config.max_box_size, 1);

    std::cout << "boxes: " << boxes.size() << "\n";

    Vector<int> procs(boxes.size(), 0);
    LoadBalance(procs, boxes);

    DisjointBoxLayout dbl(boxes, procs, base_domain);

    LevelData<FArrayBox> data(dbl, 1, IntVect::Unit);
    heat3d::problem_state soln;
    soln.alias(data);

    visit(soln,
        [&profile](Real& val, IntVect here)
        { val = profile.initial_state(here); }
    );

    typedef heat3d::imex_operators<heat3d::aniso_profile> imexop;
    ARK4<heat3d::problem_state, imexop>
        ark(imexop(profile), soln, profile.dt(), false);

    std::uint64_t step = 0;
    Real time = 0.0;
    while (time < config.nt)
    {
//        output(soln, "phi.%06u.hdf5", step); 
        data.apply([&step](Box const&, int, FArrayBox& fab) { output(fab, "phi.%06u.hdf5", step); });

        ++step;

        double const dt = profile.dt();
        ark.resetDt(dt);

        char const* fmt = "STEP %06u : TIME %.7g%|31t| += %.7g\n";
        std::cout << (boost::format(fmt) % step % time % dt) << std::flush;

        ark.advance(time, soln);

        time += dt;
    } 

    soln.exchange();
//    output(soln, "phi.%06u.hdf5", step); 
    data.apply([&step](Box const&, int, FArrayBox& fab) { output(fab, "phi.%06u.hdf5", step); });
}

