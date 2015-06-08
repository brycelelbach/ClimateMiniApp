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

#include <cstdint>
#include <chrono>

namespace climate_mini_app 
{

///////////////////////////////////////////////////////////////////////////
//
//  high_resolution_timer
//      A timer object measures elapsed time.
//
///////////////////////////////////////////////////////////////////////////
class high_resolution_timer
{
public:
    typedef std::chrono::high_resolution_clock::time_point time_point;

    high_resolution_timer()
      : start_time_(take_time_stamp())
    {
    }

    void restart()
    {
        start_time_ = take_time_stamp();
    }

    double elapsed() const // return elapsed time in seconds
    {
        using namespace std::chrono;
        auto t2 = take_time_stamp();
        return duration_cast<duration<double> >(t2 - start_time_).count();
    }

    std::int64_t elapsed_microseconds() const
    {
        using namespace std::chrono;
        auto t2 = take_time_stamp();
        return duration_cast<microseconds>(t2 - start_time_).count();
    }

    std::int64_t elapsed_nanoseconds() const
    {
        using namespace std::chrono;
        auto t2 = take_time_stamp();
        return duration_cast<nanoseconds>(t2 - start_time_).count();
    }

protected:
    static time_point take_time_stamp()
    {
        return std::chrono::high_resolution_clock::now();
    }

private:
    time_point start_time_;
};

}

#endif // CHOMBO_61F4AE03_6BFD_4C15_965B_CBEB03757036
