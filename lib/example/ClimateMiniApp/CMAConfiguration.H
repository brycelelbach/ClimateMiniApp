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

#if !defined(CHOMBO_9DCB4096_49BC_4F61_AC62_5CF9F66DB6D5)
#define CHOMBO_9DCB4096_49BC_4F61_AC62_5CF9F66DB6D5

#include "IntVect.H"
#include "StreamCSV.H"

#include <iostream>
#include <cstdint>

namespace climate_mini_app
{

enum boundary_type
{
    // Vertical
    LOWER_X,
    UPPER_X,

    // Horizontal
    LOWER_Y,
    UPPER_Y,
    LOWER_Z,
    UPPER_Z
};

enum ProblemType
{
    Problem_Invalid             = -1,
    Problem_AdvectionDiffusion  = 1,
    Problem_Last                = Problem_AdvectionDiffusion
};

inline std::ostream& operator<<(std::ostream& os, ProblemType problem)
{ // {{{
    switch (problem)
    {
        default:
        case Problem_Invalid:
            os << "Invalid";
            break;
        case Problem_AdvectionDiffusion:
            os << "Advection-Diffusion";
            break;
    }
    return os;
} // }}}

struct configuration
{
    ///////////////////////////////////////////////////////////////////////////
    // Parameters.

    ProblemType const problem;

    std::uint64_t const nh; ///< "Horizontal" extent (y and z dimensions) per core.
    std::uint64_t const nv; ///< "Vertical" extent (x dimension) per core.

    std::uint64_t const max_box_size;

    IntVect const ghost_vector;

    bool const header;  ///< Print header for CSV timing data.
    bool const verbose; ///< Print status updates.

#if defined(CH_USE_HDF5)
    bool const output; ///< Generate HDF5 output.
#endif    

    ///////////////////////////////////////////////////////////////////////////

    configuration(
        ProblemType problem_
      , std::uint64_t nh_
      , std::uint64_t nv_
      , std::uint64_t max_box_size_
      , IntVect ghost_vector_
      , bool header_
      , bool verbose_
#if defined(CH_USE_HDF5)
      , bool output_
#endif
        )
      : problem(problem_)
      , nh(nh_)
      , nv(nv_)
      , max_box_size(max_box_size_)
      , ghost_vector(ghost_vector_)
      , header(header_)
      , verbose(verbose_)
#if defined(CH_USE_HDF5)
      , output(output_)
#endif
    {}

    std::string print_csv_header() const
    { // {{{
        return "Problem,"
               "Horizontal Extent (nh),"
               "Vertical Extent (nv),"
               "Max Box Size (mbs)";
    } // }}}

    CSVTuple<ProblemType, std::uint64_t, std::uint64_t, std::uint64_t>
    print_csv() const
    { // {{{
        return StreamCSV(problem, nh, nv, max_box_size); 
    } // }}}
}; 

}

#endif // CHOMBO_9DCB4096_49BC_4F61_AC62_5CF9F66DB6D5

