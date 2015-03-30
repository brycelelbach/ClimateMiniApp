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
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/format.hpp>

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"

#include "ARK4.H"
#include "HPXDriver.H"

#include <fenv.h>
#include <assert.h>

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
 
#if defined(CH_USE_HDF5)
void output(
    AsyncLevelData<heat3d::problem_state>& data
  , std::string const& format
  , std::string const& name
  , std::uint64_t step
    )
{
    std::string file;
    try { file = boost::str(boost::format(format) % step); }
    catch (boost::io::too_many_args&) { file = format; }

    Vector<DisjointBoxLayout> dbl;
    dbl.push_back(data.disjointBoxLayout());

    Vector<std::string> names;
    names.push_back(name);

    Vector<LevelData<FArrayBox>*> level;
    level.push_back(new LevelData<FArrayBox>);
    level.back()->define(dbl[0], names.size(), data.ghostVect());

    DataIterator dit = data.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        auto& src  = data[dit()].data.data();
        auto& dest = (*level[0])[dit()];

        // Alias the src.
        dest.define(Interval(0, names.size()), src);
    }

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

    for (LevelData<FArrayBox>* ld : level) delete ld;
}
#endif

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    bool verbose_flag = vm.count("verbose");
#if defined(CH_USE_HDF5)
    bool output_flag = vm["output"].as<bool>();
#endif

    if (verbose_flag)
        std::cout << "Starting HPX/Chombo Climate Mini-App...\n"
                  << std::flush; 

    if (vm.count("header"))
        std::cout << "nt,ns,nh,nv,mbs,kx,ky,kz,Boxes,PUs,Walltime [s]\n"
                  << std::flush;

    // If both time parameters were specified, give an error.
    if (!vm["nt"].defaulted() && vm.count("ns"))
    {
        char const* fmt = "ERROR: Both --nt (=%.4g) and --ns (%u) were "
                          "specified, please provide only one time "
                          "parameter.\n"; 
        std::cout << ( boost::format(fmt)
                     % vm["nt"].as<Real>()
                     % vm["ns"].as<std::uint64_t>()) 
                  << std::flush;
        return -1;
    } 

    heat3d::configuration config(
        /*nt: physical time to step to             =*/vm["nt"].as<Real>(),
        /*nh: y and z (horizontal) extent per core =*/vm["nh"].as<std::uint64_t>(),
        /*nv: x (vertical) extent per core         =*/vm["nv"].as<std::uint64_t>(),
        /*max_box_size                             =*/vm["mbs"].as<std::uint64_t>(),
        /*ghost_vector                             =*/IntVect::Unit
    );

    heat3d::aniso_profile profile(config,
        // sine factors in the source term
        /*A=*/2.0,   /*B=*/2.0,   /*C=*/2.0,

        // diffusion coefficients
        /*kx=*/vm["kx"].as<Real>(),
        /*ky=*/vm["ky"].as<Real>(), 
        /*kz=*/vm["kz"].as<Real>()
    ); 

    if (vm.count("ns"))
    {
        // We want a multiplier slightly smaller than the desired timestep count.
        Real ns = Real(vm["ns"].as<std::uint64_t>()-1)+0.8;
        const_cast<Real&>(config.nt) = profile.dt()*ns; 
    }

    IntVect lower_bound(IntVect::Zero);
    IntVect upper_bound(
        (config.nv*numProc()-1),
        (config.nh*numProc()-1),
        (config.nh*numProc()-1)
    );

    ProblemDomain base_domain(lower_bound, upper_bound);

    Vector<Box> boxes;
    domainSplit(base_domain, boxes, config.max_box_size, 1);

    mortonOrdering(boxes);

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

#if defined(CH_USE_HDF5)
    if (output_flag)
    {
        heat3d::problem_state src; src.define(soln);  
        visit(src,
            [&profile](Real& val, IntVect here)
            { val = profile.source_term(here); }
        );
        output(src, "source.hdf5", "source", 0); 
    }
#endif 

    std::size_t step = 0;
    std::size_t epoch = 0;

    hpx::util::high_resolution_timer clock;

    Real time = 0.0;

    while (time < config.nt)
    {
        DataIterator dit = data.dataIterator();

#if defined(CH_USE_HDF5)
        Real error_sum = 0.0; 
        Real phi_sum   = 0.0; 
        Real rel_error = 0.0;

        if (output_flag)
        { 
            std::vector<hpx::future<void> > exchanges;
    
            // Fix ghost zones for output.
            for (dit.begin(); dit.ok(); ++dit)
            {
                exchanges.push_back(
                    hpx::async(LocalExchangeSync<heat3d::problem_state>
                             , epoch, dit(), HPX_STD_REF(data))
                );
            }
            for (hpx::future<void>& f : exchanges) f.get();
    
            ++epoch;
    
            output(data, "phi.%06u.hdf5", "phi_numeric", step); 
    
            AsyncLevelData<heat3d::problem_state> exact(dbl, config.ghost_vector);
            DefineData(exact, 1);
            visit(exact,
                [&profile, time](Real& val, IntVect here)
                { val = profile.exact_solution(here, time); }
            );
            output(exact, "exact.%06u.hdf5", "phi_exact", step); 
    
            for (dit.begin(); dit.ok(); ++dit)
                exact[dit()].data.increment(data[dit()].data, -1.0);
            output(exact, "error.%06u.hdf5", "error", step); 
    
            heat3d::problem_state solncopy; solncopy.copy(soln);
            exact.abs(); solncopy.abs();
            Real error_sum = exact.sum();
            Real phi_sum = solncopy.sum();
            Real rel_error = (phi_sum == 0.0 ? 0.0 : error_sum/phi_sum);
        }
#endif 

        ++step;

        double const dt = profile.dt();
        ark.resetDt(dt);

        if (verbose_flag)
        {
#if defined(CH_USE_HDF5)
            if (output)
            {
                char const* fmt = "STEP %06u : TIME %.7g%|31t| += %.7g%|43t| : "
                                  "SUM %.7g%|60t| : ERROR %.7g\n%|79t| :"
                                  "REL_ERROR %.7g\n";
                std::cout << ( boost::format(fmt) % step % time % dt
                             % phi_sum % error_sum % rel_error)
                          << std::flush;
            }

            else
#endif
            {
                char const* fmt = "STEP %06u : TIME %.7g%|31t| += %.7g\n";
                std::cout << (boost::format(fmt) % step % time % dt)
                          << std::flush;
            }
        }

        std::vector<hpx::future<void> > futures;

        for (dit.begin(); dit.ok(); ++dit)
        {
                futures.push_back(
                hpx::async(
                    [&](DataIndex di) { ark.advance(di, epoch, time, data); }
                  , dit()
                )
            );
        }

        hpx::lcos::when_all(futures).get();

        epoch += 2 * ark_type::s_nStages - 1;
        time += dt;
    } 

    std::cout << config.nt << "," 
              << step << ","
              << config.nh << ","
              << config.nv << ","
              << config.max_box_size << ","
              << profile.kx << ","
              << profile.ky << ","
              << profile.kz << ","
              << boxes.size() << ","
              << hpx::get_num_worker_threads() << ","
              << clock.elapsed() << "\n"
              << std::flush;

#if defined(CH_USE_HDF5)
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
#endif

    return 0;
}

int main(int argc, char** argv)
{
    boost::program_options::options_description cmdline("HPX/Chombo Climate Mini-App");

    cmdline.add_options()
        ( "nt"
        , boost::program_options::value<Real>()->default_value(5.0e-5, "5.0e-5")  
        , "physical time to step to")
        ( "ns"
        , boost::program_options::value<std::uint64_t>()
        , "number of steps to take")
        ( "nh"
        , boost::program_options::value<std::uint64_t>()->default_value(480)
        , "horizontal (y and z) extent per locality")
        ( "nv"
        , boost::program_options::value<std::uint64_t>()->default_value(30)
        , "vertical (x) extent per locality")
        ( "mbs"
        , boost::program_options::value<std::uint64_t>()->default_value(15)
        , "max box size")
        ( "kx"
        , boost::program_options::value<Real>()->default_value(0.25e-1, "0.25e-1")
        , "x diffusion coefficient")
        ( "ky"
        , boost::program_options::value<Real>()->default_value(1.0e-1, "1.0e-1")
        , "y diffusion coefficient")
        ( "kz"
        , boost::program_options::value<Real>()->default_value(1.0e-1, "1.0e-1")
        , "z diffusion coefficient")
        ( "header", "print the csv header")
        ( "verbose", "display status updates after each timestep")
#if defined(CH_USE_HDF5)
        ( "output"
        , boost::program_options::value<bool>()->default_value(true, "true")
        , "generate HDF5 output and sum phi every timestep") 
#endif
        ; 

    return init(chombo_main, cmdline, argc, argv); // Doesn't return
}

