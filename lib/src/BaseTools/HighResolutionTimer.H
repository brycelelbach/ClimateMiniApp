/*
 *      _______              __ 
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CHOMBO_1C771C6C_CB51_4B51_B914_D73192297D69)
#define CHOMBO_1C771C6C_CB51_4B51_B914_D73192297D69

#include <cstdint>
#include <chrono>

#include "NamespaceHeader.H"

struct HighResolutionTimer
{
    HighResolutionTimer()
      : start_time_(take_time_stamp())
    {}

    void restart()
    {
        start_time_ = take_time_stamp();
    }

    double elapsed() const // Return elapsed time in seconds.
    {
        return double(take_time_stamp() - start_time_) * 1e-9;
    }

    std::uint64_t elapsedNanoseconds() const
    {
        return take_time_stamp() - start_time_;
    }

protected:
    static std::uint64_t take_time_stamp()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now().time_since_epoch()).count();
    }

private:
    std::uint64_t start_time_;
};

#include "NamespaceFooter.H"

#endif // CHOMBO_1C771C6C_CB51_4B51_B914_D73192297D69
