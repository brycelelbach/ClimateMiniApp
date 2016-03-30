/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <mpi.h>

#if defined(CH_HPX)
    #include <hpx/config.hpp>
    #include <hpx/lcos/barrier.hpp>
    #include <hpx/runtime/actions/plain_action.hpp>
#endif

#include <boost/format.hpp>

#if !defined(CH_HPX)
    #include <boost/program_options.hpp>
#endif

#include <fenv.h>

#include "OpenMP.H"
#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"
#include "HighResolutionTimer.H"

#if defined(CH_HPX)
    #include "ARK4.H"
#else
    #include "OMPARK4.H"
#endif

#if defined(CH_HPX)
    #include "HPXDriver.H"
    #include "AsyncLevelDataRegistry.H"
#endif

#include "CMAProblemStateScratch.H"
#include "CMAAdvectionDiffusionProfile.H"

// FIXME: Move this to ALD
#if defined(CH_HPX)
template <typename LD, typename F>
void visit(LD& state, F f)
{
    std::vector<hpx::future<void> > futures;
    DataIterator dit = state.dataIterator();

    for (dit.begin(); dit.ok(); ++dit)
    { 
        auto F = 
            [&](DataIndex di)
            {
                auto& substate = state[di];
                IntVect lower = substate.smallEnd();
                IntVect upper = substate.bigEnd(); 

                for (auto k = lower[2]; k <= upper[2]; ++k)
                    for (auto j = lower[1]; j <= upper[1]; ++j)
                        for (auto i = lower[0]; i <= upper[0]; ++i)
                            f(substate(IntVect(i, j, k)), IntVect(i, j, k));
            };

        futures.emplace_back(hpx::async(F, dit()));
    }
    
    hpx::lcos::when_all(futures).get();
}
#else
template <typename LD, typename F>
void visit(LD& state, F f)
{
    DataIterator dit = state.dataIterator();
    std::size_t const nbox = dit.size();

    CH_PRAGMA_OMP(parallel for schedule(static))
    for (std::size_t ibox = 0; ibox < nbox; ++ibox)
    {
        auto& substate = state[dit[ibox]];
        IntVect lower = substate.smallEnd();
        IntVect upper = substate.bigEnd(); 

        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    f(substate(IntVect(i, j, k)), IntVect(i, j, k));
    }
}
#endif
 
#if defined(CH_USE_HDF5)
std::string format_filename(boost::format& fmter)
{
    return boost::str(fmter);
}
 
template <typename T, typename... Args>
std::string format_filename(boost::format& fmter, T&& value, Args&&... args)
{
    return format_filename(fmter % value, args...); 
}

template <typename... Args>
std::string format_filename(std::string const& fmt, Args&&... args)
{
    boost::format fmter(fmt);
    return format_filename(fmter, args...); 
}

#if defined(CH_HPX)
template <typename Profile, typename... Args>
void output(
    Profile const& profile
  , AsyncLevelData<FArrayBox>& state
  , std::string const& name
  , Real time  
  , std::string const& fmt
  , Args&&... fmt_args
    )
{
    std::string file = format_filename(fmt, fmt_args...);

    Vector<DisjointBoxLayout> level_dbl;
    level_dbl.push_back(state.disjointBoxLayout());

    Vector<std::string> names;
    names.push_back(name);

    Vector<LevelData<FArrayBox>*> level;
    level.push_back(new LevelData<FArrayBox>);
    level.back()->define(level_dbl[0], names.size(), state.ghostVect());

    DataIterator dit = state.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        auto& src  = state[dit()];
        auto& dest = (*level[0])[dit()];

        // Alias the src.
        dest.define(Interval(0, names.size()), src);
    }

    Vector<IntVect> ref_ratios;
    ref_ratios.push_back(IntVect(2, 2, 2));

    RealVect dp(
        profile.dp()[0]
      , profile.dp()[1]
      , profile.dp()[2]
    );

    WriteAnisotropicAMRHierarchyHDF5(
        file 
      , level_dbl 
      , level
      , names
      , level_dbl[0].physDomain().domainBox()
      , dp           // dx
      , profile.dt() // dt
      , time 
      , ref_ratios
      , 1 // levels
        );

    for (LevelData<FArrayBox>* ld : level) delete ld;
}
#endif

template <typename Profile, typename... Args>
void output(
    Profile const& profile
  , LevelData<FArrayBox>& state
  , std::string const& name
  , Real time  
  , std::string const& fmt
  , Args&&... fmt_args
    )
{
    std::string file = format_filename(fmt, fmt_args...);

    Vector<DisjointBoxLayout> level_dbl;
    level_dbl.push_back(state.disjointBoxLayout());

    Vector<std::string> names;
    names.push_back(name);

    Vector<IntVect> ref_ratios;
    ref_ratios.push_back(IntVect(2, 2, 2));
    
    Vector<LevelData<FArrayBox>*> level;
    level.push_back(&state); 

    RealVect dp(
        profile.dp()[0]
      , profile.dp()[1]
      , profile.dp()[2]
    );

    WriteAnisotropicAMRHierarchyHDF5(
        file 
      , level_dbl 
      , level
      , names
      , level_dbl[0].physDomain().domainBox()
      , dp           // dx
      , profile.dt() // dt
      , time 
      , ref_ratios
      , 1 // levels
        );
}
#endif

template <typename Profile>
std::pair<Real, Real> compareAndOutput(
    Profile const& profile
  , climate_mini_app::problem_state& state
  , climate_mini_app::problem_state& analytic
  , Real time
  , std::size_t step
    )
{ 
    DisjointBoxLayout const& dbl = state.U.disjointBoxLayout();

    state.exchangeSync();

    #if defined(CH_USE_HDF5) 
        if (profile.config.output)
            output(profile, state.U, "phi", time, "phi.%06u.hdf5", step); 
    #endif

    visit(analytic.U,
        [&profile, time](Real& val, IntVect here)
        { val = profile.analytic_solution(here, time); }
    );

    #if defined(CH_USE_HDF5) 
        if (profile.config.output)
            output(profile, analytic.U, "phi", time, "analytic.%06u.hdf5", step); 
    #endif


    // Compute the error, store it in analytic and find the min/max error.
    Real min_error = 1.0e300, max_error = -1.0e300;
    DataIterator dit = state.U.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        analytic.U[dit()].plus(state.U[dit()], -1.0);
        min_error = std::min(min_error, analytic.U[dit()].min());
        max_error = std::max(max_error, analytic.U[dit()].max());
    }

    #if defined(CH_USE_HDF5) 
        if (profile.config.output)
            output(profile, analytic.U, "phi", time, "error.%06u.hdf5", step); 
    #endif

    return std::pair<Real, Real>(min_error, max_error);
}

#if defined(CH_HPX)
void registration_barrier()
{
    if (numProc() == 1)
        return;

    using hpx::lcos::barrier;

    char const* const barrier_name = "/cma/barrier/registration";

    hpx::id_type here = hpx::find_here();

    if (hpx::find_root_locality() == here)
    {
        // Create the barrier, register it with AGAS
        hpx::lcos::barrier b = hpx::lcos::barrier::create(here, numProc());
        hpx::agas::register_name_sync(barrier_name, b.get_gid());
        b.wait();
        hpx::agas::unregister_name_sync(barrier_name);
    }

    else
    {
        hpx::id_type id = hpx::agas::on_symbol_namespace_event(
                barrier_name, hpx::agas::symbol_ns_bind, true).get();
        hpx::lcos::barrier b(id);
        b.wait();
    }
}
#endif

template <typename Profile>
void stepLoop(
    Profile const& profile
  , Real nt
    )
{ 
    HighResolutionTimer clock;

    climate_mini_app::configuration const& config = profile.config;

    auto const procid = procID();

    if (config.verbose && (0 == procid))
        std::cout << "Starting Chombo Climate Mini-App...\n"
                  << std::flush; 

    ProblemDomain base_domain = profile.problem_domain(); 

    Vector<Box> boxes;
    domainSplit(base_domain, boxes, config.max_box_size, 1);

    mortonOrdering(boxes);

    Vector<int> procs(boxes.size(), 0);
    LoadBalance(procs, boxes);

    if (config.verbose && (0 == procid))
    {
        std::cout << "Processor Assignment:\n";

        for (auto i = 0; i < boxes.size(); ++i) 
            std::cout << "  " << boxes[i] << " -> " << procs[i] << "\n";
    }

    DisjointBoxLayout dbl(boxes, procs, base_domain);

    climate_mini_app::problem_state
        state(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_STATE);

    #if defined(CH_HPX)
        RegisterALDSync(&(state.U));
    #endif

    visit(state.U,
        [&profile](Real& val, IntVect here)
        { val = profile.initial_state(here); }
    );

    climate_mini_app::problem_state
        analytic(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_ANALYTIC);

    #if defined(CH_HPX)
        RegisterALDSync(&(analytic.U));
    #endif

    #if defined(CH_HPX)
        typedef ARK4<
            climate_mini_app::problem_state_fab
          , climate_mini_app::problem_state_fab_scratch<false>
          , Profile
          , false
        > ark_type;
    
        std::array<climate_mini_app::problem_state, ark_type::s_nStages> phi;
    
        phi[0].define(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_PHI0);
        phi[1].define(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_PHI1);
        phi[2].define(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_PHI2);
        phi[3].define(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_PHI3);
        phi[4].define(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_PHI4);
        phi[5].define(dbl, 1, profile.ghostVect(), config.tile_width, climate_mini_app::PS_TAG_PHI5);

        RegisterALDSync(&(phi[0].U));
        RegisterALDSync(&(phi[1].U));
        RegisterALDSync(&(phi[2].U));
        RegisterALDSync(&(phi[3].U));
        RegisterALDSync(&(phi[4].U));
        RegisterALDSync(&(phi[5].U));
 
        // FIXME: Remove these.
        climate_mini_app::problem_state kE(dbl, 1, profile.ghostVect(), config.tile_width);
        climate_mini_app::problem_state kI(dbl, 1, profile.ghostVect(), config.tile_width);

        registration_barrier();
    #else
        typedef OMPARK4<
            climate_mini_app::problem_state
          , climate_mini_app::problem_state_scratch<false>
          , Profile
        > ark_type;
  
        ark_type ark(profile, profile.dt()); 
        ark.define(dbl, 1, profile.ghostVect(), config.tile_width);
    #endif

    std::size_t step = 0;

    double init_elapsed = clock.elapsed();

    clock.restart();

    Real time = 0.0;

    while ((time < nt) && (std::fabs(nt-time) > 1e-14))
    {
        DataIterator dit = state.U.dataIterator();

        #if defined(CH_USE_HDF5) 
            if (config.output)
                compareAndOutput(profile, state, analytic, time, step);
        #endif

        ++step;

        double const dt = profile.dt();

        #if !defined(CH_HPX)
            ark.resetDt(dt);
        #endif

        if (config.verbose && (0 == procid))
        {
            char const* fmt = "STEP %06u : TIME %.7g%|31t| += %.7g\n";
            std::cout << (boost::format(fmt) % step % time % dt)
                      << std::flush;
        }

        #if defined(CH_HPX)
            std::vector<hpx::future<void> > futures;
    
            for (dit.begin(); dit.ok(); ++dit)
            {
                auto advance =  
                    [&](DataIndex di)
                    {
                        climate_mini_app::problem_state_fab substate(state, di);
    
                        ark_type ark(profile, dt); 
                        ark.define(di, phi, kE, kI);
    
                        ark.advance(time, substate);
                    };
    
                futures.emplace_back(hpx::async(advance, dit()));
            }
    
            hpx::lcos::when_all(futures).get();
        #else
            ark.advance(time, state);
        #endif

        #if defined(CH_USE_HDF5) && defined(CH_DUMP_FLUXES)
            if (config.output)
            { 
                for (std::size_t stage = 0; stage < ark_type::s_nStages; ++stage) 
                {
                    Real s_time = time + ark_type::s_c[stage]*dt;
                    std::size_t s_step = ark_type::s_nStages*(step-1) + stage;
    
                    auto& FY = phi[stage].FY;
                    auto& FZ = phi[stage].FZ;
    
                    output(profile, FY, "phi", s_time, "FY.%06u.hdf5", s_step); 
                    output(profile, FZ, "phi", s_time, "FZ.%06u.hdf5", s_step); 
                }
            }
        #endif

        time += dt;
    } 

    std::uint64_t localities = numProc(); 
    #if defined(CH_HPX)
        std::uint64_t pus = hpx::get_os_thread_count() * localities; 
    #else
        std::uint64_t pus = omp_get_max_threads() * localities;
    #endif

    std::pair<Real, Real> min_max_error
        = compareAndOutput(profile, state, analytic, time, step);

    if (0 == procid)
    {
        if (config.header)
            std::cout << config.print_csv_header() << "," 
                      << profile.print_csv_header() << ","
                      << "Boxes,"
                         "Steps (ns),"
                         "Simulation Time (nt),"
                         "Localities,"
                         "PUs,"
                         "Minimum Error,"
                         "Maximum Error,"
                         "Initialization Wall Time [s],"
                         "Solver Wall Time [s]\n"
                      << std::flush;

        std::cout << config.print_csv() << "," 
                  << profile.print_csv() << ","
                  << boxes.size() << ","
                  << step << ","
                  << nt << ","
                  << localities << ","
                  << pus << ","
                  << min_max_error.first << ","
                  << min_max_error.second << ","
                  << init_elapsed << "," 
                  << clock.elapsed() << "\n"
                  << std::flush;
    }
}

#if defined(CH_HPX)
HPX_PLAIN_ACTION(stepLoop<climate_mini_app::advection_diffusion_profile>
               , cma_step_loop_action);
#endif

using boost::program_options::variables_map;

int chombo_main(variables_map& vm)
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    ///////////////////////////////////////////////////////////////////////////
    // Command-line processing

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
        return 1;
    } 

    ///////////////////////////////////////////////////////////////////////////
    // Configuration

    climate_mini_app::configuration config(
        /*nh: x and y (horizontal) extent          =*/ vm["nh"].as<std::uint64_t>()
        /*nv: z (vertical) extent                  =*/,vm["nv"].as<std::uint64_t>()
        /*max_box_size                             =*/,vm["mbs"].as<std::uint64_t>()
        /*tw: width in y dimension of each tile    =*/,vm["tw"].as<std::uint64_t>()
        /*header: print header for CSV timing data =*/,vm.count("header")
        /*verbose: print status updates            =*/,vm.count("verbose") 
        #if defined(CH_USE_HDF5)
        /*output: generate HDF5 output             =*/,vm["output"].as<bool>()
        #endif
    );

    typedef climate_mini_app::advection_diffusion_profile profile_type;

    profile_type profile(config,
        /*cx=*/vm["cx"].as<Real>(),
        /*cy=*/vm["cy"].as<Real>(),
        /*cz=*/vm["cz"].as<Real>(),

        // velocity components
        /*vx=*/vm["vx"].as<Real>(), 
        /*vy=*/vm["vy"].as<Real>(),

        // diffusion coefficients
        /*kz=*/vm["kz"].as<Real>()
    ); 

    Real nt = vm["nt"].as<Real>();

    // Correct nt if --ns was specified.
    if (vm.count("ns"))
    {
        // We want a multiplier slightly smaller than the desired
        // timestep count.
        Real ns = Real(vm["ns"].as<std::uint64_t>()-1)+0.8;
        nt = profile.dt()*ns; 
    }

    #if defined(CH_HPX)
    {
        auto clients = CreateAsyncLevelDataRegistry<FArrayBox>();

        assert(clients.size() == numProc());

        std::vector<hpx::future<void> > futures;
        futures.reserve(clients.size());

        for (std::size_t i = 0; i < clients.size(); ++i)
        {
            auto gid = hpx::naming::get_locality_from_id(clients[i].get_gid());
            futures.emplace_back(
                hpx::async<cma_step_loop_action>(gid, profile, nt));
        }

//        for (hpx::future<void>& f : futures)
//            f.get();
        hpx::lcos::when_all(futures).get();
    }
    #else
        stepLoop(profile, nt);
    #endif

    return 0;
}

int main(int argc, char** argv)
{
    boost::program_options::options_description cmdline(
        "HPX/Chombo AMR Climate Mini-App"
        );

    cmdline.add_options()
        ( "nt"
        , boost::program_options::value<Real>()->
            default_value(0.5, "0.5")  
        , "physical time to step to")
        ( "ns"
        , boost::program_options::value<std::uint64_t>()
        , "number of steps to take")
        ( "nh"
        , boost::program_options::value<std::uint64_t>()->
            default_value(480)
        , "horizontal (y and z) extent per locality")
        ( "nv"
        , boost::program_options::value<std::uint64_t>()->
            default_value(30)
        , "vertical (x) extent per locality")
        ( "mbs"
        , boost::program_options::value<std::uint64_t>()->
            default_value(30)
        , "max box size")
        ( "tw"
        , boost::program_options::value<std::uint64_t>()->
            default_value(2)
        , "width in y dimension of each tile")

        ( "cx"
        , boost::program_options::value<Real>()->
            default_value(2.0, "2.0")
        , "x constant")
        ( "cy"
        , boost::program_options::value<Real>()->
            default_value(2.0, "2.0")
        , "y constant")
        ( "cz"
        , boost::program_options::value<Real>()->
            default_value(2.0, "2.0")
        , "z constant")

        ( "vx"
        , boost::program_options::value<Real>()->
            default_value(1.0e-1, "1.0e-1")
        , "x velocity component")
        ( "vy"
        , boost::program_options::value<Real>()->
            default_value(1.0e-1, "1.0e-1")
        , "y velocity component")

        ( "kz"
        , boost::program_options::value<Real>()->
            default_value(1.0e-2, "1.0e-2")
        , "z diffusion coefficient")

        ( "header", "print header for the CSV timing data")
        ( "verbose", "display status updates")

        #if defined(CH_USE_HDF5)
        ( "output"
        , boost::program_options::value<bool>()->
            default_value(true, "true")
        , "generate HDF5 output every timestep") 
        #endif
        ; 

    #if defined(CH_HPX)
        return init(chombo_main, cmdline, argc, argv); // Doesn't return
    #else
        int requested_threading_model = MPI_THREAD_FUNNELED;
        int actual_threading_model = -1;

        MPI_Init_thread(&argc, &argv
                      , requested_threading_model, &actual_threading_model);

        assert(requested_threading_model == actual_threading_model);

        cmdline.add_options()
            ( "threads,t"
            , boost::program_options::value<std::uint64_t>()->
                default_value(1)
            , "number of OS-threads")
            ( "help", "print this information")
            ;
 
        boost::program_options::variables_map vm;
    
        boost::program_options::store(
            boost::program_options::command_line_parser
                (argc, argv).options(cmdline).run(), vm);
    
        boost::program_options::notify(vm);
    
        // Print help screen.
        if (vm.count("help"))
        {
            std::cout << cmdline;
            std::exit(0);
        }

        std::uint64_t num_threads = vm["threads"].as<std::uint64_t>();

        omp_set_num_threads(num_threads);

        int r = chombo_main(vm);

        MPI_Finalize();

        return r;
    #endif
}

