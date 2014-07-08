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

void output(FArrayBox const& soln, std::string const& format, std::string const& name, std::uint64_t step)
{
    std::string file;
    try { file = boost::str(boost::format(format) % step); }
    catch (boost::io::too_many_args&) { file = format; }

    writeFABname(&soln, file.c_str(), Vector<std::string>(1, name));
}

void output(LevelData<FArrayBox> const& data, std::string const& format, std::string const& name, std::uint64_t step)
{
    data.apply([format, name, step](Box const&, int, FArrayBox& fab) { output(fab, format, name, step); });
}

void output(heat3d::problem_state const& soln, std::string const& format, std::string const& name, std::uint64_t step)
{
//    std::string file = boost::str(boost::format(format) % step);
//    writeLevelname(&soln.data(), file.c_str());
    output(soln.data(), format, name, step);
}

int main()
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    heat3d::configuration config(
        /*nt: physical time to step to             =*/0.005,
        /*nh: y and z (horizontal) extent per core =*/60,
        /*nv: x (vertical) extent per core         =*/30,
        /*max_box_size                             =*/60
    );

    heat3d::aniso_profile profile(config,
        // sine factors in the source term
        /*A=*/2.0,   /*B=*/2.0,   /*C=*/2.0,

        // diffusion coefficients
        /*kx=*/0.25, /*ky=*/1.0, /*kz=*/1.0
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

    LevelData<FArrayBox> data(dbl, 1, IntVect::Zero);
    heat3d::problem_state soln;
    soln.alias(data);

    visit(soln,
        [&profile](Real& val, IntVect here)
        { val = profile.initial_state(here); }
    );

    typedef heat3d::imex_operators<heat3d::aniso_profile> imexop;
    ARK4<heat3d::problem_state, imexop>
        ark(imexop(profile), soln, profile.dt(), false);

    heat3d::problem_state src; src.define(soln);  
    visit(src,
        [&profile](Real& val, IntVect here)
        { val = profile.source_term(here); }
    );
    output(src, "source.hdf5", "source", 0); 

    std::uint64_t step = 0;
    Real time = 0.0;
    while (time < config.nt)
    {
        output(data, "phi.%06u.hdf5", "phi_numeric", step); 

        heat3d::problem_state exact; exact.define(soln);
        visit(exact,
            [&profile, time](Real& val, IntVect here)
            { val = profile.exact_solution(here, time); }
        );
        output(exact, "exact.%06u.hdf5", "phi_exact", step); 

        exact.increment(soln, -1.0); 
        output(exact, "error.%06u.hdf5", "error", step); 

//        heat3d::problem_state solncopy; solncopy.copy(soln);
//        exact.abs(); solncopy.abs();
//        Real error_sum = exact.sum();
//        Real phi_sum = solncopy.sum();
//        Real rel_error = (phi_sum == 0.0 ? 0.0 : error_sum/phi_sum);

        ++step;

        double const dt = profile.dt();
        ark.resetDt(dt);

        char const* fmt = "STEP %06u : "
                          "TIME %.7g%|31t| += %.7g%|43t| : "
                          "SUM %.7g%|60t| : ERROR %.7g\n";
        std::cout << (boost::format(fmt) % step % time % dt % soln.sum() % exact.sum()) << std::flush;

        ark.advance(time, soln);

        time += dt;
    } 

    soln.exchange();
    output(soln, "phi.%06u.hdf5", "phi_numeric", step); 

    heat3d::problem_state exact; exact.define(soln); 
    visit(exact,
        [&profile, time](Real& val, IntVect here)
        { val = profile.exact_solution(here, time); }
    );
    output(exact, "exact.%06u.hdf5", "phi_exact", step); 

    exact.increment(soln, -1.0); 
    output(exact, "error.%06u.hdf5", "error", step); 
}

