/*
 *      _______              __ 
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c) 2014-2015 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_61F4AE03_6BFD_4C15_965B_CBEB03757036)
#define CHOMBO_61F4AE03_6BFD_4C15_965B_CBEB03757036

#include <boost/cstdint.hpp>
#include <boost/chrono/chrono.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>

namespace climate_mini_app 
{
    
struct high_resolution_clock
{
    // This function returns a tick count with a resolution (not
    // precision!) of 1 ns.
    static boost::uint64_t now()
    {
        boost::chrono::nanoseconds ns =
            boost::chrono::steady_clock::now().time_since_epoch();
        return static_cast<boost::uint64_t>(ns.count());
    }

    // This function returns the smallest representable time unit as
    // returned by this clock.
    static boost::uint64_t (min)()
    {
        typedef boost::chrono::duration_values<boost::chrono::nanoseconds>
            duration_values;
        return (duration_values::min)().count();
    }

    // This function returns the largest representable time unit as
    // returned by this clock.
    static boost::uint64_t (max)()
    {
        typedef boost::chrono::duration_values<boost::chrono::nanoseconds>
            duration_values;
        return (duration_values::max)().count();
    }
};

///////////////////////////////////////////////////////////////////////////
//
//  high_resolution_timer
//      A timer object measures elapsed time.
//
///////////////////////////////////////////////////////////////////////////
class high_resolution_timer
{
public:
    high_resolution_timer()
      : start_time_(take_time_stamp())
    {
    }

    high_resolution_timer(double t)
      : start_time_(static_cast<boost::uint64_t>(t * 1e9))
    {}

    static double now()
    {
        return take_time_stamp() * 1e-9;
    }

    void restart()
    {
        start_time_ = take_time_stamp();
    }
    double elapsed() const                  // return elapsed time in seconds
    {
        return double(take_time_stamp() - start_time_) * 1e-9;
    }

    boost::int64_t elapsed_microseconds() const
    {
        return boost::int64_t((take_time_stamp() - start_time_) * 1e-3);
    }

    boost::int64_t elapsed_nanoseconds() const
    {
        return boost::int64_t(take_time_stamp() - start_time_);
    }

    double elapsed_max() const   // return estimated maximum value for elapsed()
    {
        return (high_resolution_clock::max)() * 1e-9;
    }

    double elapsed_min() const   // return minimum value for elapsed()
    {
        return (high_resolution_clock::min)() * 1e-9;
    }

protected:
    static boost::uint64_t take_time_stamp()
    {
        return high_resolution_clock::now();
    }

private:
    boost::uint64_t start_time_;
};

}

#endif // CHOMBO_61F4AE03_6BFD_4C15_965B_CBEB03757036
