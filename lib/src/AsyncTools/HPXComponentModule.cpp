////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012-2014 Bryce Adelstein-Lelbach aka wash
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// This TU is the only game in town for HPX component registeration.

#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "HPXServices.H"
#include "BoxLayout.H"
#include "FArrayBox.H"
#include "AsyncLevelDataRegistry.H"
//#include "DistributedAsyncLevelData.H"

#include "NamespaceVar.H"

HPX_REGISTER_COMPONENT_MODULE();

///////////////////////////////////////////////////////////////////////////////

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::simple_component<CH_XD::HPXServicesServer>,
    chombo_ServicesServer);

///////////////////////////////////////////////////////////////////////////////

// Add function-style overloading for the registeration macro using a forwarding
// variadic macro.
#define HPX_CHOMBO_REGISTER_DALDR_ACTIONS(...)                              \
        HPX_CHOMBO_REGISTER_DALDR_(__VA_ARGS__)                             \
    /**/

#define HPX_CHOMBO_REGISTER_DALDR_(...)                                     \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                          \
        HPX_CHOMBO_REGISTER_DALDR_,                                         \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                   \
    )(__VA_ARGS__))                                                         \
    /**/

// One argument overload, uses the typename as the registeration name.
#define HPX_CHOMBO_REGISTER_DALDR_1(T)                                      \
    HPX_CHOMBO_REGISTER_DALDR_2(T, T)                                       \
    /**/

// Two argument overload, uses a user-specified name for registeration. This
// version must be used if the type T is not alphanumeric (e.g. if it's a
// template, or namespace qualified). 
#define HPX_CHOMBO_REGISTER_DALDR_2(T, name)                                \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                                 \
        hpx::components::managed_component<                                 \
            CH_XD::AsyncLevelDataRegistryServer<T>                          \
        >,                                                                  \
        chombo_AsyncLevelDataRegistryServer_ ## name                        \
    );                                                                      \
    HPX_REGISTER_ACTION(                                                    \
        CH_XD::AsyncLevelDataRegistryServer<T>::createDispatcher_action,    \
        chombo_AsyncLevelDataRegistryServer_createDispatcher_ ## name       \
    );                                                                      \
    HPX_REGISTER_ACTION(                                                    \
        CH_XD::AsyncLevelDataRegistryServer<T>::insert_action,              \
        chombo_AsyncLevelDataRegistryServer_insert_ ## name                 \
    );                                                                      \
    HPX_REGISTER_ACTION(                                                    \
        CH_XD::AsyncLevelDataRegistryServer<T>::apply_action,               \
        chombo_AsyncLevelDataRegistryServer_apply_ ## name                  \
    );                                                                      \
    /**/

HPX_CHOMBO_REGISTER_DALDR_ACTIONS(CH_XD::FArrayBox, FArrayBox); 

