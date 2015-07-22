/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012-2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_50E1AEAC_4E1D_4F10_8A0B_F08E6FED219C)
#define CHOMBO_50E1AEAC_4E1D_4F10_8A0B_F08E6FED219C

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/naming/name.hpp>

#include "FArrayBox.H"

#include "AsyncLevelData.H"

#include "NamespaceHeader.H"

template <typename T>
struct AsyncLevelDataServer;

template <typename T, typename enable = void>
struct AsyncLevelData_traits;

template <typename T, typename enable>
struct AsyncLevelData_traits
{
    typedef typename T::value_type value_type;
    typedef T mesh_type; 
};

template <typename T, typename X>
struct AsyncLevelData_signature;

template <typename T, typename X>
struct AsyncLevelData_signature
{
    typedef typename AsyncLevelData_traits<T>::value_type value_type;
    typedef typename AsyncLevelData_traits<T>::mesh_type mesh_type;
    typedef AsyncLevelDataServer<T> ald_type;

    typedef hpx::util::function<
        // grid
        X(ald_type&)
    > gridwise;

    typedef hpx::util::function<
        // grid, mesh-index, mesh
        X(ald_type&, DataIndex, mesh_type&)
    > boxwise;

    typedef hpx::util::function<
        // grid, mesh-index, mesh, element-index, element
        X(ald_type&, DataIndex, mesh_type&, IntVect, value_type&)
    > elementwise;

    typedef hpx::util::function<
        X(X const&, X const&)
    > reducer;
};

template <typename T>
struct AsyncLevelData_signature<T, void>
{
    typedef typename AsyncLevelData_traits<T>::value_type value_type;
    typedef typename AsyncLevelData_traits<T>::mesh_type mesh_type;
    typedef AsyncLevelDataServer<T> ald_type;

    typedef hpx::util::function<
        // grid
        void(ald_type&)
    > gridwise;

    typedef hpx::util::function<
        // grid, mesh-index, mesh
        void(ald_type&, DataIndex, mesh_type&)
    > meshwise;

    typedef hpx::util::function<
        // grid, mesh-index, mesh, element-index, element
        void(ald_type&, DataIndex, mesh_type&, IntVect, value_type&)
    > elementwise;
};

template <typename T>
struct AsyncLevelDataServer
  : hpx::components::managed_component_base<AsyncLevelDataServer<T> >
{
    typedef AsyncLevelDataServer this_type;

    typedef typename AsyncLevelData_traits<T>::value_type value_type;
    typedef typename AsyncLevelData_traits<T>::mesh_type mesh_type;

    AsyncLevelData<T> local_data;

    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X apply(
        typename AsyncLevelData_signature<T, X>::gridwise const& f
        )
    { // {{{
        return f(*this);
    } // }}}

    template <typename X>
    struct apply_action
      : hpx::actions::make_action<
            X (this_type::*)(
                typename AsyncLevelData_signature<T, X>::gridwise const&
            )
          , &this_type::template apply<X>
          , apply_action<X>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    void applyElements(
        typename AsyncLevelData_signature<T, void>::elementwise const& f
        )
    { // {{{
        DataIterator dit(local_data.dataIterator());
        for (dit.begin(); dit.ok(); ++dit)
        { 
            DataIndex mi = dit();
            mesh_type& m = local_data[mi];

            IntVect lower = m.smallEnd();
            IntVect upper = m.bigEnd(); 

            for (auto k = lower[2]; k <= upper[2]; ++k)
                for (auto j = lower[1]; j <= upper[1]; ++j)
                    for (auto i = lower[0]; i <= upper[0]; ++i)
                    {
                        IntVect ei(i, j, k); 
                        value_type& e = m(ei);
                        f(*this, mi, m, ei, e); 
                    }
        }
    } // }}}

    HPX_DEFINE_COMPONENT_ACTION(this_type,
                                applyElements,
                                applyElements_action); 

/*
    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X applyBox(
        DataIndex di
      , hpx::util::function<X(mesh_type&)> const& f
        );

    template <typename X>
    struct applyBox_action
      : hpx::actions::make_action<
            X (this_type::*)
                ( DataIndex
                , hpx::util::function<X(mesh_type&)> const&)
          , &this_type::template applyBox<X>
          , applyBox_action<X>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X applyBoxElements(
        DataIndex di
      , hpx::util::function<X(value_type&)> const& f
        );

    template <typename X>
    struct applyBoxElements_action
      : hpx::actions::make_action<
            X (this_type::*)
                ( DataIndex
                , hpx::util::function<X(value_type&)> const&)
          , &this_type::template applyBoxElements<X>
          , applyBoxElements_action<X>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X reduce(
        hpx::util::function<X(this_type&)> const& f
      , hpx::util::function<X(X const&, X const&)> const& reducer
      , X const& initial = X()
        );

    template <typename X>
    struct reduce_action
      : hpx::actions::make_action<
            X (this_type::*)
                ( hpx::util::function<X(this_type&)> const&
                , hpx::util::function<X(X const&, X const&)> const&
                , X const&)
          , &octree_server::template reduce<X>
          , reduce_action<X>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X reduceElements(
        hpx::util::function<X(value_type&)> const& f
      , hpx::util::function<X(X const&, X const&)> const& reducer
      , X const& initial = X()
        );

    template <typename X>
    struct reduceElements_action
      : hpx::actions::make_action<
            X (this_type::*)
                ( hpx::util::function<X(value_type&)> const&
                , hpx::util::function<X(X const&, X const&)> const&
                , X const&)
          , &octree_server::template reduceElements<X>
          , reduceElements_action<X>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X reduceBox(
        DataIndex di
      , hpx::util::function<X(mesh_type&)> const& f
      , hpx::util::function<X(X const&, X const&)> const& reducer
      , X const& initial = X()
        );

    template <typename X>
    struct reduceBox_action
      : hpx::actions::make_action<
            X (this_type::*)
                ( DataIndex
                , hpx::util::function<X(mesh_type&)> const&
                , hpx::util::function<X(X const&, X const&)> const&
                , X const&)
          , &octree_server::template reduceBox<X>
          , reduceBox_action<X>
        >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    X reduceBoxElements(
        DataIndex di
      , hpx::util::function<X(value_type&)> const& f
      , hpx::util::function<X(X const&, X const&)> const& reducer
      , X const& initial = X()
        );

    template <typename X>
    struct reduceBoxElements_action
      : hpx::actions::make_action<
            X (this_type::*)
                ( DataIndex
                , hpx::util::function<X(value_type&)> const&
                , hpx::util::function<X(X const&, X const&)> const&
                , X const&)
          , &octree_server::template reduceBoxElements<X>
          , reduceBoxElements_action<X>
        >
    {};
*/
};

template <typename T>
struct AsyncLevelDataClient
  : hpx::components::client_base<AsyncLevelDataClient<T>, AsyncLevelDataServer<T> >
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename X>
    hpx::future<X> applyAsync(
        typename AsyncLevelData_signature<T, X>::gridwise const& f
        )
    { // {{{
        typedef typename AsyncLevelDataServer<T>::template apply_action<X>
            action;
        return hpx::async<action>(this->get_gid(), f);
    } // }}}

    template <typename X>
    X apply(
        typename AsyncLevelData_signature<T, X>::gridwise const& f
        )
    { // {{{
        return applyAsync(f);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void applyElementsAsync(
        typename AsyncLevelData_signature<T, void>::elementwise const& f
        )
    { // {{{
        typedef typename AsyncLevelDataServer<T>::applyElements
            action;
        return hpx::async<action>(this->get_gid(), f);
    } // }}}

    void applyElements(
        typename AsyncLevelData_signature<T, void>::elementwise const& f
        )
    { // {{{
        return applyElementsAsync(f);
    } // }}}
};

#include "NamespaceFooter.H"

#include "NamespaceVar.H"

// Add function-style overloading for the registeration macro using a forwarding
// variadic macro.
#define HPX_CHOMBO_REGISTER_ALD_DECLARATION(...)                            \
        HPX_CHOMBO_REGISTER_ALD_DECLARATION_(__VA_ARGS__)                   \
    /**/

#define HPX_CHOMBO_REGISTER_ALD_DECLARATION_(...)                           \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                          \
        HPX_CHOMBO_REGISTER_ALD_DECLARATION_,                               \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                   \
    )(__VA_ARGS__))                                                         \
    /**/

// One argument overload, uses the typename as the registeration name.
#define HPX_CHOMBO_REGISTER_ALD_DECLARATION_1(T)                            \
    HPX_CHOMBO_REGISTER_ALD_DECLARATION_2(T, T)                             \
    /**/

// Two argument overload, uses a user-specified name for registeration. This
// version must be used if the type T is not alphanumeric (e.g. if it's a
// template, or namespace qualified). 
#define HPX_CHOMBO_REGISTER_ALD_DECLARATION_2(T, name)                      \
    HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(                               \
        (template <typename X>),                                            \
        (CH_XD::AsyncLevelDataServer<T>::apply_action<X>)                   \
    );                                                                      \
    HPX_REGISTER_ACTION_DECLARATION(                                        \
        CH_XD::AsyncLevelDataServer<T>::applyElements_action,               \
        chombo_AsyncLevelDataServer_applyElements_ ## name                  \
    );                                                                      \
    /**/

HPX_CHOMBO_REGISTER_ALD_DECLARATION(CH_XD::FArrayBox, FArrayBox); 

#endif // CHOMBO_50E1AEAC_4E1D_4F10_8A0B_F08E6FED219C
