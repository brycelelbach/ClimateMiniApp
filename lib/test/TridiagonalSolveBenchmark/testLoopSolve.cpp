#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <boost/program_options.hpp>

#include <omp.h>

#include "FArrayBox.H"
#include "LevelData.H"
#include "LayoutIterator.H"
#include "LoadBalance.H"
#include "BRMeshRefine.H"
#include "HighResolutionTimer.H"

#include "TestData.H"
#include "TestImExOp.H"

#include "UsingNamespace.H"

// Check that the data is constant, and return the constant
Real ldfabVal(LevelData<FArrayBox>& a_data, Real tolerance)
{
    Real ldmin = CH_BADVAL;
    Real ldmax = -CH_BADVAL;

    DisjointBoxLayout dbl = a_data.getBoxes();

    DataIterator dit(dbl);
    for (dit.begin(); dit.ok(); ++dit)
    {
        Box b = dbl[dit];
        Real min = a_data[dit].min(b);
        ldmin = (ldmin < min) ? ldmin : min;
        Real max = a_data[dit].max(b);
        ldmax = (ldmax > max) ? ldmax : max;
    }

    if (std::fabs(ldmax - ldmin) > tolerance)
    {
        std::cout << "Min/max values of error diff more than tolerance: "
                  << tolerance << "\n" 
                  << "  min: " << ldmin << ", max: " << ldmax << "\n";
    }

    return std::max(std::fabs(ldmax), std::fabs(ldmin));
}

int testImExBE(boost::program_options::variables_map& vm)
{
    std::uint64_t const ns  = vm["ns"].as<std::uint64_t>();

    std::uint64_t const nh  = vm["nh"].as<std::uint64_t>();
    std::uint64_t const nv  = vm["nv"].as<std::uint64_t>();
    std::uint64_t const mbs = vm["mbs"].as<std::uint64_t>();
    std::uint64_t const tw  = vm["tw"].as<std::uint64_t>();

    Real const tolerance    = vm["tolerance"].as<Real>();

    bool const header       = vm.count("header");

    if (0 == procID())
    {
        if (0 != (nh % 8))
        {
            std::cout << "ERROR: nh (" << nh << ") must be a multiple of 8\n";
            return 2;
        }

        if (nv > mbs)
        {
            std::cout << "ERROR: nv (" << nv << ") must be less than or equal "
                         "to mbs (" << mbs << ")\n";
            return 2;
        }

        if (0 != (mbs % 8))
        {
            std::cout << "ERROR: mbs (" << mbs << ") must be a multiple of 8\n";
            return 2;
        }

        if (0 != ((mbs + 8) % tw))
        {
            std::cout << "ERROR: mbs (" << mbs << ") + 8 must be a divisible "
                         "by tw (" << tw << ")\n";
            return 2;
        }
    }

    if (header && (0 == procID()))
        std::cout << "Steps (ns),"
                     "Simulation Time,"
                     "Horizontal Extent (nh),"
                     "Vertical Extent (nv),"
                     "Max Box Size (mbs),"
                     "Tile Width (tw),"
                     "Boxes,"
                     "Tiles,"
                     "Localities,"
                     "PUs,"
                     "Error,"
                     "Tolerance,"
                     "Execution Time [s]\n";

    IntVect const num_cells(nh, nh, nv);
    IntVect const lo_vect = IntVect::Zero;
    IntVect const hi_vect = num_cells - IntVect::Unit;
    Box const domain_box(lo_vect, hi_vect);
    ProblemDomain const base_domain(domain_box);

    Vector<Box> boxes;
    domainSplit(base_domain, boxes, mbs, 1);

    Vector<int> procs(boxes.size(), 0);
    LoadBalance(procs, boxes);

    DisjointBoxLayout dbl(boxes, procs, base_domain);

    IntVect const ghosts(4, 4, 0);
    TestData soln(dbl, 1, ghosts, tw);
    TestData exact(dbl, 1, ghosts, tw);

    Real time = 0.0;
    Real const base_dt = std::sqrt(0.5);
    Real const dt = base_dt / Real(ns);

    TestImExOp imex_op(dt);

    // Initial conditions.
    imex_op.exact(soln, time);

    HighResolutionTimer t;

    for (long step = 0; step < ns; ++step)
    {
        time = imex_op.advance(soln, time);
    }

    double exec_time = t.elapsed();

    // Calculate the error.
    imex_op.exact(exact, time);

    exact.plus(soln, -1.0);

    Real const error = std::fabs(ldfabVal(exact.U, tolerance));

    bool passes = (error <= tolerance);

    if (0 == procID())
    {
        std::uint64_t const num_localities = numProc(); 
        std::uint64_t const num_pus = omp_get_max_threads() * num_localities;

        std::uint64_t const num_boxes = boxes.size();
        std::uint64_t const num_tiles = ((mbs + 8) / tw) * num_boxes;
 
        std::cout << ns << ","
                  << time << ","
                  << nh << ","
                  << nv << "," 
                  << mbs << "," 
                  << tw << "," 
                  << num_boxes << "," 
                  << num_tiles << "," 
                  << num_localities << "," 
                  << num_pus << "," 
                  << error << "," 
                  << tolerance << "," 
                  << exec_time << "\n";
    }

    return (passes) ? 0 : 1;
}

int main(int argc ,char* argv[])
{
    boost::program_options::options_description cmdline(
        "Vectorized Vertical Solve Benchmark"
        );

    cmdline.add_options()
        ( "ns"
        , boost::program_options::value<std::uint64_t>()->
            default_value(10)
        , "number of steps to take")

        ( "nh"
        , boost::program_options::value<std::uint64_t>()->
            default_value(112)
        , "horizontal (x and y) extent per locality (must be a multiple of 8)")
        ( "nv"
        , boost::program_options::value<std::uint64_t>()->
            default_value(32)
        , "vertical (z) extent per locality (must be less than or equal to mbs)")
        ( "mbs"
        , boost::program_options::value<std::uint64_t>()->
            default_value(56)
        , "max box size (must be a multiple of 8)")
        ( "tw"
        , boost::program_options::value<std::uint64_t>()->
            default_value(32)
        , "number of x rows per tile (mbs + 8 must be divisible by tw)")

        ( "tolerance"
        , boost::program_options::value<Real>()->
            default_value(2.0e-14, "2.0e-14")
        , "error tolerance")

        ( "threads,t"
        , boost::program_options::value<std::uint64_t>()->default_value(1)
        , "number of OS-threads")

        ( "header", "print header for the CSV timing data")

        ( "help", "print this information")
        ; 

    #if defined(CH_MPI)
        int requested_threading_model = MPI_THREAD_FUNNELED;
        int actual_threading_model = -1;

        MPI_Init_thread(&argc, &argv
                      , requested_threading_model, &actual_threading_model);

        assert(requested_threading_model == actual_threading_model);
    #endif

    boost::program_options::variables_map vm;
    
    boost::program_options::store(
        boost::program_options::command_line_parser
            (argc, argv).options(cmdline).run(), vm);
    
    boost::program_options::notify(vm);
    
    // Print help.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        std::exit(0);
    }

    std::uint64_t num_threads = vm["threads"].as<std::uint64_t>();

    omp_set_num_threads(num_threads);

    int status = testImExBE(vm);

    #if defined(CH_MPI)
        MPI_Finalize();
    #endif

    return status;
}

