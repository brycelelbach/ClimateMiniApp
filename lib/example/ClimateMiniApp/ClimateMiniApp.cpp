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

#if defined(CH_HPX)
    #include <hpx/config.hpp>
    #include <hpx/util/high_resolution_timer.hpp>
#endif

#include <boost/format.hpp>

#if !defined(CH_HPX)
    #include <boost/program_options.hpp>
#endif

#include <fenv.h>

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"

#include "ARK4.H"

#if defined(CH_HPX)
    #include "HPXDriver.H"
#endif

#include "CMAProblemStateScratch.H"
#include "CMAAdvectionDiffusionProfile.H"
#include "CMAHighResolutionTimer.H"

// FIXME: Move this to ALD
template <typename LD, typename F>
void visit(LD& soln, F f)
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

template <typename Profile, typename... Args>
void output(
    Profile const& profile
  , AsyncLevelData<FArrayBox>& data
  , std::string const& name
  , Real time  
  , std::string const& fmt
  , Args&&... fmt_args
    )
{
    std::string file = format_filename(fmt, fmt_args...);

    Vector<DisjointBoxLayout> level_dbl;
    level_dbl.push_back(data.disjointBoxLayout());

    Vector<std::string> names;
    names.push_back(name);

    Vector<LevelData<FArrayBox>*> level;
    level.push_back(new LevelData<FArrayBox>);
    level.back()->define(level_dbl[0], names.size(), data.ghostVect());

    DataIterator dit = data.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        auto& src  = data[dit()];
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

template <typename Profile, typename... Args>
void output(
    Profile const& profile
  , LevelData<FArrayBox>& data
  , std::string const& name
  , Real time  
  , std::string const& fmt
  , Args&&... fmt_args
    )
{
    std::string file = format_filename(fmt, fmt_args...);

    Vector<DisjointBoxLayout> level_dbl;
    level_dbl.push_back(data.disjointBoxLayout());

    Vector<std::string> names;
    names.push_back(name);

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
      , data
      , names
      , level_dbl[0].physDomain().domainBox()
      , dp           // dx
      , profile.dt() // dt
      , time 
      , ref_ratios
      , 1 // levels
        );
}

template <typename Profile>
void compareAndOutput(
    Profile const& profile
  , climate_mini_app::problem_state& data
  , Real time
  , std::size_t step
    )
{ 
    DisjointBoxLayout const& dbl = data.U.disjointBoxLayout();

    data.exchangeSync();

    output(profile, data.U, "phi", time, "phi.%06u.hdf5", step); 

    climate_mini_app::problem_state analytic(dbl, 1, data.U.ghostVect());
    visit(analytic.U,
        [&profile, time](Real& val, IntVect here)
        { val = profile.analytic_solution(here, time); }
    );

    output(profile, analytic.U, "phi", time, "analytic.%06u.hdf5", step); 

    DataIterator dit = data.U.dataIterator();

    for (dit.begin(); dit.ok(); ++dit)
        analytic.U[dit()].plus(data.U[dit()], -1.0);
    output(profile, analytic.U, "phi", time, "error.%06u.hdf5", step); 
}
#endif

template <typename Profile>
void stepLoop(
    Profile const& profile
  , Real nt
    )
{ 
    climate_mini_app::configuration const& config = profile.config;

    if (config.verbose)
        std::cout << "Starting HPX/Chombo Climate Mini-App...\n"
                  << std::flush; 

    if (config.header)
        std::cout << config.print_csv_header() << "," 
                  << profile.print_csv_header() << ","
                  << "Boxes,PUs,Steps,Simulation Time,Wall Time [s]\n"
                  << std::flush;

    ProblemDomain base_domain = profile.problem_domain(); 

    Vector<Box> boxes;
    domainSplit(base_domain, boxes, config.max_box_size, 1);

    mortonOrdering(boxes);

    Vector<int> procs(boxes.size(), 0);
    LoadBalance(procs, boxes);

    DisjointBoxLayout dbl(boxes, procs, base_domain);

    climate_mini_app::problem_state data(dbl, 1, config.ghost_vector);

    visit(data.U,
        [&profile](Real& val, IntVect here)
        { val = profile.initial_state(here); }
    );

    #if defined(CH_HPX)
        typedef climate_mini_app::imex_per_box_operators<Profile> imexop;
        typedef ARK4<
            climate_mini_app::sub_problem_state
          , climate_mini_app::sub_problem_state_scratch<false>
          , imexop
          , false
        > ark_type;
    
        std::array<climate_mini_app::problem_state, ark_type::s_nStages> phi;
    
        for (climate_mini_app::problem_state& ps : phi)
            ps.define(dbl, 1, config.ghost_vector);
    
        // FIXME: Can we remove flux, ghost zones?
        climate_mini_app::problem_state kE(dbl, 1, config.ghost_vector);
        climate_mini_app::problem_state kI(dbl, 1, config.ghost_vector);
    #else
        typedef climate_mini_app::imex_per_ld_operators<Profile> imexop;
        typedef ARK4<
            climate_mini_app::problem_state
          , climate_mini_app::problem_state_scratch<false>
          , imexop
          , false
        > ark_type;
  
        ark_type ark; 
        ark.define(dbl, 1, config.ghost_vector);
    #endif

    #if defined(CH_USE_HDF5) && defined(CH_DUMP_SOURCE_TERM)
        if (config.output)
        {
            climate_mini_app::problem_state src(dbl, 1, config.ghost_vector);
            visit(src.U,
                [&profile](Real& val, IntVect here)
                { val = profile.source_term(here, 0.0); }
            );
            output(profile, src.U, "phi", 0.0, "source.hdf5"); 
        }
    #endif 

    std::size_t step = 0;

    climate_mini_app::high_resolution_timer clock;

    Real time = 0.0;

    while ((time < nt) && (std::fabs(nt-time) > 1e-14))
    {
        DataIterator dit = data.U.dataIterator();

        #if defined(CH_USE_HDF5)
            if (config.output)
                compareAndOutput(profile, data, time, step);
        #endif 

        ++step;

        double const dt = profile.dt();

        if (config.verbose)
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
                        climate_mini_app::sub_problem_state subdata(data, di);
    
                        ark_type ark(imexop(profile), dt); 
                        ark.define(di, phi, kE, kI);
    
                        ark.advance(time, subdata);
                    };
    
                futures.emplace_back(hpx::async(advance, dit()));
            }
    
            hpx::lcos::when_all(futures).get();
        #else
            ark.advance(time, data);
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

    std::cout << config.print_csv() << "," 
              << profile.print_csv() << ","
              << boxes.size() << ","
              << hpx::get_num_worker_threads() << ","
              << step << ","
              << nt << ","
              << clock.elapsed() << "\n"
              << std::flush;

    #if defined(CH_USE_HDF5)
        if (config.output)
            compareAndOutput(profile, data, time, step);
    #endif
}

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

    std::string const problem_str = vm["problem"].as<std::string>();

    climate_mini_app::ProblemType problem = climate_mini_app::Problem_Invalid;    

    if ("advection-diffusion" == problem_str)
        problem = climate_mini_app::Problem_AdvectionDiffusion;
    else
    {
        char const* fmt = "ERROR: Invalid argument provided to "
                          "--problem (=%s), current problem(s): "
                          "'advection-diffusion'\n"; 
        std::cout << (boost::format(fmt) % problem_str)
                  << std::flush;
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Configuration

    climate_mini_app::configuration config(
        /*problem: type of problem                 =*/problem, 
        /*nh: y and z (horizontal) extent per core =*/vm["nh"].as<std::uint64_t>(),
        /*nv: x (vertical) extent per core         =*/vm["nv"].as<std::uint64_t>(),
        /*max_box_size                             =*/vm["mbs"].as<std::uint64_t>(),
        #if defined(CH_LOWER_ORDER_EXPLICIT_STENCIL)
        /*ghost_vector                             =*/IntVect::Unit,
        #else
        /*ghost_vector                             =*/IntVect::Unit*3,
        #endif
        /*header: print header for CSV timing data =*/vm.count("header"),
        /*verbose: print status updates            =*/vm.count("verbose"), 
        #if defined(CH_USE_HDF5)
        /*output: generate HDF5 output             =*/vm["output"].as<bool>()
        #endif
    );

    if (climate_mini_app::Problem_AdvectionDiffusion == config.problem)
    {
        typedef climate_mini_app::advection_diffusion_profile profile_type;

        profile_type profile(config,
            /*cx=*/vm["cx"].as<Real>(),
            /*cy=*/vm["cy"].as<Real>(),
            /*cz=*/vm["cz"].as<Real>(),

            // diffusion coefficients
            /*kx=*/vm["kx"].as<Real>(),

            // velocity components
            /*ky=*/vm["vy"].as<Real>(), 
            /*kz=*/vm["vz"].as<Real>()
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

        stepLoop(profile, nt);
    }

    // Invalid problem type.
    else
        assert(false);

    return 0;
}

int main(int argc, char** argv)
{
    boost::program_options::options_description cmdline(
        "HPX/Chombo AMR Climate Mini-App"
        );

    cmdline.add_options()
        ( "problem"
        , boost::program_options::value<std::string>()->
            default_value("advection-diffusion")
        , "type of problem (options: advection-diffusion)") 

        ( "nt"
        , boost::program_options::value<Real>()->
            default_value(5.0e-5, "5.0e-5")  
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
            default_value(15)
        , "max box size")

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

        ( "kx"
        , boost::program_options::value<Real>()->
            default_value(1.0e-2, "1.0e-2")
        , "x diffusion coefficient")

        ( "vy"
        , boost::program_options::value<Real>()->
            default_value(1.0e-1, "1.0e-1")
        , "y velocity component")
        ( "vz"
        , boost::program_options::value<Real>()->
            default_value(1.0e-1, "1.0e-1")
        , "z velocity component")

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

        return chombo_main(vm);
    #endif
}

