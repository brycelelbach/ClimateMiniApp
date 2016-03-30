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

#include <cstdint>
#include <iostream>

#include "IntVect.H"
#include "StreamCSV.H"
#include "BoostSerialization.H"

namespace climate_mini_app
{

struct configuration
{
    ///////////////////////////////////////////////////////////////////////////
    // Parameters.

    std::uint64_t nh; ///< "Horizontal" extent (x and y dimensions) per core.
    std::uint64_t nv; ///< "Vertical" extent (z dimension) per core.

    std::uint64_t max_box_size;

    std::uint64_t tile_width;

    bool header;  ///< Print header for CSV timing data.
    bool verbose; ///< Print status updates.

    #if defined(CH_USE_HDF5)
    bool output; ///< Generate HDF5 output.
    #endif    

    ///////////////////////////////////////////////////////////////////////////
    configuration()
      : nh()
      , nv()
      , max_box_size()
      , tile_width()
      , header()
      , verbose()
      #if defined(CH_USE_HDF5)
      , output()
      #endif
    {}

    configuration(
        std::uint64_t nh_
      , std::uint64_t nv_
      , std::uint64_t max_box_size_
      , std::uint64_t tile_width_
      , bool header_
      , bool verbose_
      #if defined(CH_USE_HDF5)
      , bool output_
      #endif
        )
      : nh(nh_)
      , nv(nv_)
      , max_box_size(max_box_size_)
      , tile_width(tile_width_)
      , header(header_)
      , verbose(verbose_)
      #if defined(CH_USE_HDF5)
      , output(output_)
      #endif
    {}

    std::string print_csv_header() const
    { // {{{
        return "Horizontal Extent (nh),"
               "Vertical Extent (nv),"
               "Max Box Size (mbs)",
               "Tile Width (tw)";
    } // }}}

    CSVTuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>
    print_csv() const
    { // {{{
        return StreamCSV(nh, nv, max_box_size, tile_width); 
    } // }}}

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    { // {{{
        ar & nh;
        ar & nv;
        ar & max_box_size;
        ar & tile_width;
        ar & header;
        ar & verbose;
        #if defined(CH_USE_HDF5)
        ar & output;
        #endif
    } // }}}
}; 

}

#endif // CHOMBO_9DCB4096_49BC_4F61_AC62_5CF9F66DB6D5

