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

#include "AsyncARK4.H"
#include "HPXDriver.H"

#include <fenv.h>

#include "transport_3d.hpp"

template <typename F>
void visit(AsyncLevelData<FArrayBox>& soln, F f)
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
                    f(subsoln(IntVect(i, j, k)), IntVect(i, j, k));
    }
}
 
#if defined(CH_USE_HDF5)
void output(
    AsyncLevelData<FArrayBox>& data
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
        auto& src  = data[dit()];
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
        std::cout << "nt,ns,nh,nv,mbs,kx,ky,kz,vy,vz,Boxes,PUs,Walltime [s]\n"
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

    transport_3d::configuration config(
        /*nt: physical time to step to             =*/vm["nt"].as<Real>(),
        /*nh: y and z (horizontal) extent per core =*/vm["nh"].as<std::uint64_t>(),
        /*nv: x (vertical) extent per core         =*/vm["nv"].as<std::uint64_t>(),
        /*max_box_size                             =*/vm["mbs"].as<std::uint64_t>(),
        /*ghost_vector                             =*/IntVect::Unit
    );

    transport_3d::aniso_profile profile(config,
        // sine factors in the source term
        /*A=*/2.0,   /*B=*/2.0,   /*C=*/2.0,

        // diffusion coefficients
        /*kx=*/vm["kx"].as<Real>(),
        /*ky=*/vm["ky"].as<Real>(), 
        /*kz=*/vm["kz"].as<Real>(),

        // velocity components
        /*ky=*/vm["vy"].as<Real>(), 
        /*kz=*/vm["vz"].as<Real>()
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

    bool is_periodic[] = { true, true, true };
    ProblemDomain base_domain(lower_bound, upper_bound, is_periodic);

    Vector<Box> boxes;
    domainSplit(base_domain, boxes, config.max_box_size, 1);

    mortonOrdering(boxes);

    Vector<int> procs(boxes.size(), 0);
    LoadBalance(procs, boxes);

    DisjointBoxLayout dbl(boxes, procs, base_domain);

    transport_3d::problem_state data(dbl, 1, config.ghost_vector);

    visit(data.U,
        [&profile](Real& val, IntVect here)
        { val = profile.initial_state(here); }
    );

    typedef transport_3d::imex_operators<transport_3d::aniso_profile> imexop;
    typedef AsyncARK4<transport_3d::problem_state, imexop> ark_type;
 
    ark_type ark(imexop(profile), dbl, 1, config.ghost_vector, profile.dt(), false);

#if defined(CH_USE_HDF5)
    if (output_flag)
    {
        transport_3d::problem_state src(dbl, 1, config.ghost_vector);
        visit(src.U,
            [&profile](Real& val, IntVect here)
            { val = profile.source_term(here, 0.0); }
        );
        output(src.U, "source.hdf5", "phi", 0); 
    }
#endif 

    std::size_t step = 0;
    std::size_t global_epoch = 0;

    hpx::util::high_resolution_timer clock;

    Real time = 0.0;

    while (time < config.nt)
    {
        DataIterator dit = data.U.dataIterator();

#if defined(CH_USE_HDF5)
        if (output_flag)
        { 
            std::vector<hpx::future<void> > exchanges;
    
            // Fix ghost zones for output.
            for (dit.begin(); dit.ok(); ++dit)
            {
                exchanges.push_back(
                    hpx::async(LocalExchangeSync<FArrayBox>
                             , global_epoch, dit(), std::ref(data.U))
                );
            }

            hpx::lcos::when_all(exchanges).get();
    
            ++global_epoch;
    
            output(data.U, "phi.%06u.hdf5", "phi", step); 
    
            transport_3d::problem_state analytic(dbl, 1, config.ghost_vector);
            visit(analytic.U,
                [&profile, time](Real& val, IntVect here)
                { val = profile.analytic_solution(here, time); }
            );
            output(analytic.U, "analytic.%06u.hdf5", "phi", step); 
    
            for (dit.begin(); dit.ok(); ++dit)
                analytic.increment(dit(), data, -1.0);
            output(analytic.U, "error.%06u.hdf5", "phi", step); 
        }
#endif 

        ++step;

        double const dt = profile.dt();
        ark.resetDt(dt);

        if (verbose_flag)
        {
            char const* fmt = "STEP %06u : TIME %.7g%|31t| += %.7g\n";
            std::cout << (boost::format(fmt) % step % time % dt)
                      << std::flush;
        }

        std::vector<hpx::future<void> > futures;

        for (dit.begin(); dit.ok(); ++dit)
        {
            futures.emplace_back(
                hpx::async(
                    [&](std::size_t epoch, DataIndex di)
                    { ark.advance(epoch, di, time, data); }
                  , global_epoch, dit()
                )
            );
        }

        hpx::lcos::when_all(futures).get();

        global_epoch += 2 * ark_type::s_nStages - 1;
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
              << profile.vy << ","
              << profile.vz << ","
              << boxes.size() << ","
              << hpx::get_num_worker_threads() << ","
              << clock.elapsed() << "\n"
              << std::flush;

#if defined(CH_USE_HDF5)
    if (output_flag)
    { 
        DataIterator dit = data.U.dataIterator();

        std::vector<hpx::future<void> > exchanges;
    
        // Fix ghost zones for output.
        for (dit.begin(); dit.ok(); ++dit)
        {
            exchanges.push_back(
                hpx::async(LocalExchangeSync<FArrayBox>
                         , global_epoch, dit(), std::ref(data.U))
            );
        }

        hpx::lcos::when_all(exchanges).get();

        output(data.U, "phi.%06u.hdf5", "phi", step); 
    
        transport_3d::problem_state analytic(dbl, 1, config.ghost_vector);
        visit(analytic.U,
            [&profile, time](Real& val, IntVect here)
            { val = profile.analytic_solution(here, time); }
        );

        output(analytic.U, "analytic.%06u.hdf5", "phi", step); 
    
        for (dit.begin(); dit.ok(); ++dit)
            analytic.increment(dit(), data, -1.0);
        output(analytic.U, "error.%06u.hdf5", "phi", step); 
    }
#endif

    return 0;
}

int main(int argc, char** argv)
{
    boost::program_options::options_description cmdline(
        "HPX/Chombo AMR Climate Mini-App (Transport Problem)"
        );

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
        , boost::program_options::value<Real>()->default_value(0.25e-2, "0.25e-2")
        , "x diffusion coefficient")
        ( "ky"
        , boost::program_options::value<Real>()->default_value(1.0e-2, "1.0e-2")
        , "y diffusion coefficient")
        ( "kz"
        , boost::program_options::value<Real>()->default_value(1.0e-2, "1.0e-2")
        , "z diffusion coefficient")

        ( "vy"
        , boost::program_options::value<Real>()->default_value(1.0e-1, "1.0e-1")
        , "y velocity component")
        ( "vz"
        , boost::program_options::value<Real>()->default_value(1.0e-1, "1.0e-1")
        , "z velocity component")

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

