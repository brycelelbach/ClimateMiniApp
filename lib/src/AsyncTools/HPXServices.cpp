/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-14 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-14 Hartmut Kaiser 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/include/lcos.hpp>

#include "HPXServices.H"

#include "NamespaceHeader.H"

HPXServicesServer* runtime_ptr = 0;

HPXServicesServer::HPXServicesServer(
    std::vector<hpx::id_type> const& localities
    )
  : localities_(localities) 
{
    if (runtime_ptr != 0)
        HPX_THROW_IN_CURRENT_FUNC(hpx::invalid_status,
            "runtime_ptr has already been set");

    runtime_ptr = this;
}

///////////////////////////////////////////////////////////////////////////////

void call_here(hpx::util::function<void()> const& f) 
{
    f();  
}

#include "NamespaceFooter.H"

HPX_PLAIN_ACTION(CH_XD::call_here, chombo_call_here_action);

#include "NamespaceHeader.H"

std::vector<hpx::future<void> > call_everywhere(
    hpx::util::function<void()> const& f
    ) 
{
    if (localities().empty())
        HPX_THROW_IN_CURRENT_FUNC(hpx::assertion_failure,
            "no localities supporting Chombo's HPX services available");

    std::vector<hpx::future<void> > calls;
    calls.reserve(localities().size());

    for (boost::uint64_t i = 0; i < localities().size(); ++i)
        calls.emplace_back(
            hpx::async<chombo_call_here_action>(localities()[i], f));

    return calls;
}

#include "NamespaceFooter.H"

