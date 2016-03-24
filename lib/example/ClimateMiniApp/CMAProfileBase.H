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

#if !defined(CHOMBO_834AE853_D8B6_487B_8B2C_6C97EE5E7151)
#define CHOMBO_834AE853_D8B6_487B_8B2C_6C97EE5E7151

#include <array>

#include "CMAConfiguration.H"

namespace climate_mini_app
{

template <typename Derived>
struct profile_base 
{
  public:
    profile_base()
    // {{{
      : config()
    {}
    // }}}

    profile_base(configuration config_)
    // {{{
      : config(config_)
    {}
    // }}}

    IntVect ghostVect() const
    { // {{{
        return config.ghost_vector;
    } // }}}

    std::array<Real, 3> face_coords(IntVect here) const
    { // {{{
        std::array<Real, 3> c;
        c[0] = Real(here[0])/(Real(config.nv));
        c[1] = Real(here[1])/(Real(config.nh));
        c[2] = Real(here[2])/(Real(config.nh));
        return c;
    } // }}}

    std::array<Real, 3> center_coords(IntVect here) const
    { // {{{
        std::array<Real, 3> c;
        c[0] = face_coords(here)[0]+0.5*dp()[0];
        c[1] = face_coords(here)[1]+0.5*dp()[1];
        c[2] = face_coords(here)[2]+0.5*dp()[2];
        return c;
    } // }}}

    // Spatial step size
    std::array<Real, 3> dp() const
    { // {{{
        return face_coords(IntVect(1, 1, 1));
    } // }}}

    bool is_outside_domain(boundary_type bdry, int l) const
    { // {{{
        switch (bdry)
        {
            // Vertical
            case LOWER_X:
                return (l <= -1);
            case UPPER_X:
                return (l >= config.nv);

            // Horizontal
            case LOWER_Y:
            case LOWER_Z:
                return (l <= -1); 
            case UPPER_Y:
            case UPPER_Z:
                return (l >= config.nh);
        };

        assert(false);
        return false;
    } // }}} 

    bool is_outside_domain(IntVect here) const
    { // {{{
        return is_outside_domain(LOWER_X, here[0])
            || is_outside_domain(LOWER_Y, here[1])
            || is_outside_domain(LOWER_Z, here[2])
            || is_outside_domain(UPPER_X, here[0])
            || is_outside_domain(UPPER_Y, here[1])
            || is_outside_domain(UPPER_Z, here[2]);
    } // }}}

    bool is_outside_domain(Real x, Real y, Real z) const
    { // {{{
        return is_outside_domain(LOWER_X, x)
            || is_outside_domain(LOWER_Y, y)
            || is_outside_domain(LOWER_Z, z)
            || is_outside_domain(UPPER_X, x)
            || is_outside_domain(UPPER_Y, y)
            || is_outside_domain(UPPER_Z, z);
    } // }}}

    bool is_boundary(boundary_type bdry, int l) const
    { // {{{
        switch (bdry)
        {
            // Vertical
            case LOWER_X:
                return (l == 0);
            case UPPER_X:
                return (l == config.nv-1);

            // Horizontal
            case LOWER_Y:
            case LOWER_Z:
                return (l == 0);
            case UPPER_Y:
            case UPPER_Z:
                return (l == config.nh-1);
        };

        assert(false);
        return false; 
    } // }}}

    bool is_boundary(IntVect here) const
    { // {{{
        return is_boundary(LOWER_X, here[0])
            || is_boundary(LOWER_Y, here[1])
            || is_boundary(LOWER_Z, here[2])
            || is_boundary(UPPER_X, here[0])
            || is_boundary(UPPER_Y, here[1])
            || is_boundary(UPPER_Z, here[2]);
    } // }}}

    bool is_boundary(Real x, Real y, Real z) const
    { // {{{
        return is_boundary(LOWER_X, x)
            || is_boundary(LOWER_Y, y)
            || is_boundary(LOWER_Z, z)
            || is_boundary(UPPER_X, x)
            || is_boundary(UPPER_Y, y)
            || is_boundary(UPPER_Z, z);
    } // }}}

    configuration config;

  private:
    Derived const& derived() const
    { // {{{
        return static_cast<Derived const&>(*this);
    } // }}}
};

}

#endif // CHOMBO_834AE853_D8B6_487B_8B2C_6C97EE5E7151

