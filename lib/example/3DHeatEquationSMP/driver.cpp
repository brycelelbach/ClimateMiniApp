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

#include <hpx/config.hpp>

#include <boost/format.hpp>

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"

#include "ARK4.H"
#include "HPXDriver.H"

#include <fenv.h>

#include "heat3d.hpp"

template <typename F>
void visit(AsyncLevelData<heat3d::problem_state>& soln, F f)
{
    DataIterator dit = soln.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    { 
        auto& subsoln = soln[dit].data.data();
        IntVect lower = subsoln.smallEnd();
        IntVect upper = subsoln.bigEnd(); 

        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    f(subsoln(IntVect(i, j, k)), IntVect(i, j, k));
    }
}

/*
void output(AsyncLevelData<FArrayBox> const& data, std::string const& format, std::string const& name, std::uint64_t step)
{
    std::string file;
    try { file = boost::str(boost::format(format) % step); }
    catch (boost::io::too_many_args&) { file = format; }

    Vector<DisjointBoxLayout> dbl;
    dbl.push_back(data.disjointBoxLayout());

    Vector<LevelData<FArrayBox>*> level;
    level.push_back(const_cast<LevelData<FArrayBox>*>(&data));

    Vector<std::string> names;
    names.push_back(name);

    Vector<int> ref_ratios;
    ref_ratios.push_back(2);
    ref_ratios.push_back(2);

    WriteAMRHierarchyHDF5(
        file 
      , dbl 
      , level
      , names
      , dbl[0].physDomain().domainBox() 
      , 1.0 // dx
      , 1.0 // dt
      , 0.0 // time
      , ref_ratios
      , 1 // levels
        );
}
*/

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    heat3d::configuration config(
        /*nt: physical time to step to             =*/0.005,
        /*nh: y and z (horizontal) extent per core =*/60,
        /*nv: x (vertical) extent per core         =*/30,
        /*max_box_size                             =*/15,
        /*ghost_vector                             =*/IntVect::Unit
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

    AsyncLevelData<heat3d::problem_state> data(dbl, IntVect::Unit);
    DefineData(data, 1);

    visit(data,
        [&profile](Real& val, IntVect here)
        { val = profile.initial_state(here); }
    );

    typedef heat3d::imex_operators<heat3d::aniso_profile> imexop;
    typedef ARK4<heat3d::problem_state, imexop> ark_type;
 
    ark_type ark(imexop(profile), dbl, 1, data.ghostVect(), profile.dt(), false);

/*
    heat3d::problem_state src; src.define(soln);  
    visit(src,
        [&profile](Real& val, IntVect here)
        { val = profile.source_term(here); }
    );
    output(src, "source.hdf5", "source", 0); 
*/

    std::size_t step = 0;
    std::size_t epoch = 0;

    Real time = 0.0;
    while (time < config.nt)
    {
//        soln.exchange();
//        output(soln, "phi.%06u.hdf5", "phi_numeric", step); 

/*
        heat3d::problem_state exact; exact.define(soln);
        visit(exact,
            [&profile, time](Real& val, IntVect here)
            { val = profile.exact_solution(here, time); }
        );
        output(exact, "exact.%06u.hdf5", "phi_exact", step); 

        exact.increment(soln, -1.0); 
        output(exact, "error.%06u.hdf5", "error", step); 
*/

//        heat3d::problem_state solncopy; solncopy.copy(soln);
//        exact.abs(); solncopy.abs();
//        Real error_sum = exact.sum();
//        Real phi_sum = solncopy.sum();
//        Real rel_error = (phi_sum == 0.0 ? 0.0 : error_sum/phi_sum);

        ++step;

        double const dt = profile.dt();
        ark.resetDt(dt);

/*
        char const* fmt = "STEP %06u : "
                          "TIME %.7g%|31t| += %.7g%|43t| : "
                          "SUM %.7g%|60t| : ERROR %.7g\n";
        std::cout << (boost::format(fmt) % step % time % dt % soln.sum() % exact.sum()) << std::flush;
*/
        char const* fmt = "STEP %06u : "
                          "TIME %.7g%|31t| += %.7g%|43t|\n";
        std::cout << (boost::format(fmt) % step % time % dt) << std::flush;

        std::vector<hpx::future<void> > futures;

        DataIterator dit = data.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            futures.push_back(
                hpx::async(
                    [&](DataIndex di) { ark.advance(di, epoch, time, data); }
                  , dit()
                )
            );
        }

        for (hpx::future<void>& f : futures) f.get();

        epoch += 2 * ark_type::s_nStages - 1;
        time += dt;
    } 

/*
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
*/
    return 0;
}

int main(int argc, char** argv)
{
    return init(chombo_main, argc, argv); // Doesn't return
}

