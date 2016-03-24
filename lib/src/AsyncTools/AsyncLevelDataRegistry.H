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

#if !defined(CHOMBO_2CC46AF8_3D69_4CC8_947E_269835EC7B60)
#define CHOMBO_2CC46AF8_3D69_4CC8_947E_269835EC7B60

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/naming/name.hpp>

#include "FArrayBox.H"

#include "AsyncLevelData.H"

#include "NamespaceHeader.H"

template <typename T>
struct AsyncLevelDataRegistryServer;

template <typename T>
struct AsyncLevelDataRegistryClient;

template <typename T>
struct AsyncLevelDataRegistryDispatcher;

///////////////////////////////////////////////////////////////////////////////
// Interface

// Call once per T.
template <typename T>
std::vector<AsyncLevelDataRegistryClient<T> > CreateAsyncLevelDataRegistry();

///////////////////////////////////////////////////////////////////////////////

template <typename T>
hpx::future<void> RegisterALDAsync(
    AsyncLevelData<T>* local_data
    );

template <typename T>
void RegisterALDSync(
    AsyncLevelData<T>* local_data
    );

template <typename T>
hpx::future<void> ApplyALDAsync(
    std::uint32_t proc
  , std::size_t tag
  , hpx::util::function<void(AsyncLevelData<T>&)> const& f
    );

template <typename T>
void ApplyALDSync(
    std::uint32_t proc
  , std::size_t tag
  , hpx::util::function<void(AsyncLevelData<T>&)> const& f
    );

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct AsyncLevelDataRegistryServer
  : hpx::components::managed_component_base<AsyncLevelDataRegistryServer<T> >
{
    typedef AsyncLevelDataRegistryServer this_type;

    std::map<std::size_t, AsyncLevelData<T>* > local_data;

    ///////////////////////////////////////////////////////////////////////////
    void createDispatcher(
        std::vector<AsyncLevelDataRegistryClient<T> > const& clients
        );

    HPX_DEFINE_COMPONENT_ACTION(this_type,
                                createDispatcher,
                                createDispatcher_action); 

    ///////////////////////////////////////////////////////////////////////////
    void insert(
        std::size_t tag
      , std::uintptr_t local_ptr
        )
    { // {{{
        assert(0 == local_data.count(tag));
        assert(0 != local_ptr); 

        local_data[tag] = reinterpret_cast<AsyncLevelData<T>*>(local_ptr);
    } // }}}

    HPX_DEFINE_COMPONENT_ACTION(this_type,
                                insert,
                                insert_action); 

    ///////////////////////////////////////////////////////////////////////////
    void apply(
        std::size_t tag
      , hpx::util::function<void(AsyncLevelData<T>&)> const& f
        )
    { // {{{
        assert(1 == local_data.count(tag));

        f(*local_data[tag]);
    } // }}}

    HPX_DEFINE_COMPONENT_ACTION(this_type,
                                apply,
                                apply_action); 
};

template <typename T>
struct AsyncLevelDataRegistryClient
  : hpx::components::client_base<
        AsyncLevelDataRegistryClient<T>, AsyncLevelDataRegistryServer<T>
    >
{
    typedef hpx::components::client_base<
        AsyncLevelDataRegistryClient<T>, AsyncLevelDataRegistryServer<T>
    > base_type;

    AsyncLevelDataRegistryClient()
      : base_type()
    {}

    AsyncLevelDataRegistryClient(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {}

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<void> createDispatcherAsync(
        std::vector<AsyncLevelDataRegistryClient<T> > const& clients
        )
    { // {{{
        typedef typename
            AsyncLevelDataRegistryServer<T>::createDispatcher_action action;
        return hpx::async<action>(this->get_gid(), clients);
    } // }}}

    void createDispatcherSync(
        std::vector<AsyncLevelDataRegistryClient<T> > const& clients
        )
    { // {{{
        createDispatcher(clients).get();
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<void> insertAsync(
        AsyncLevelData<T>* local_data
        )
    { // {{{
        typedef typename AsyncLevelDataRegistryServer<T>::insert_action action;
        std::uintptr_t local_ptr = reinterpret_cast<std::uintptr_t>(local_data);
        return hpx::async<action>(this->get_gid(), local_data->tag(), local_ptr);
    } // }}}

    void insertSync(
        AsyncLevelData<T>* local_data
        )
    { // {{{
        insertAsync(local_data).get();
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<void> applyAsync(
        std::size_t tag
      , hpx::util::function<void(AsyncLevelData<T>&)>&& f
        )
    { // {{{
        typedef typename AsyncLevelDataRegistryServer<T>::apply_action action;
        return hpx::async<action>(this->get_gid(), tag, f);
    } // }}}

    void applySync(
        std::size_t tag
      , hpx::util::function<void(AsyncLevelData<T>&)>&& f
        )
    { // {{{
        applyAsync(tag, std::move(f)).get();
    } // }}}
};

template <typename T>
struct AsyncLevelDataRegistryDispatcher
{
  private:
    static AsyncLevelDataRegistryDispatcher* this_;

    std::map<std::uint32_t, AsyncLevelDataRegistryClient<T> > clients;

  public:
    AsyncLevelDataRegistryDispatcher(
        std::vector<AsyncLevelDataRegistryClient<T> > const& clients_
        )
    { // {{{
        assert(clients.empty());

        for (auto const& client : clients_)
        {
            clients[hpx::naming::get_locality_id_from_id(client.get_gid())]
                = client;
        }
    } // }}}

    static void create(
        std::vector<AsyncLevelDataRegistryClient<T> > const& clients_
        )
    { // {{{
        // Make sure we do not initialize twice.
        assert(0 == this_);

        this_ = new AsyncLevelDataRegistryDispatcher<T>(clients_);
    } // }}}

    static hpx::future<void> insertAsync(
        AsyncLevelData<T>* local_data
        )
    { // {{{
        std::uint32_t proc =
            hpx::naming::get_locality_id_from_id(hpx::find_here());

        assert(this_);
        assert(1 == this_->clients.count(proc));

        return this_->clients[proc].insertAsync(local_data);
    } // }}}

    static void insertSync(
        AsyncLevelData<T>* local_data
        )
    { // {{{
        insertAsync(local_data).get();
    } // }}}

    static hpx::future<void> applyAsync(
        std::uint32_t proc
      , std::size_t tag
      , hpx::util::function<void(AsyncLevelData<T>&)>&& f
        )
    { // {{{
        assert(this_);
        assert(1 == this_->clients.count(proc));

        return this_->clients[proc].applyAsync(tag, std::move(f));
    } // }}}

    static void applySync(
        std::uint32_t proc
      , std::size_t tag
      , hpx::util::function<void(AsyncLevelData<T>&)>&& f
        )
    { // {{{
        applyAsync(proc, tag, std::move(f)).get();
    } // }}}
};

template<typename T>
AsyncLevelDataRegistryDispatcher<T>*
AsyncLevelDataRegistryDispatcher<T>::this_ = 0;

template <typename T>
void AsyncLevelDataRegistryServer<T>::createDispatcher(
    std::vector<AsyncLevelDataRegistryClient<T> > const& clients
    )
{ // {{{
    AsyncLevelDataRegistryDispatcher<T>::create(clients);
} // }}}

// Call once per T.
template <typename T>
std::vector<AsyncLevelDataRegistryClient<T> > CreateAsyncLevelDataRegistry()
{ // {{{
    hpx::components::component_type type =
        hpx::components::get_component_type<AsyncLevelDataRegistryServer<T> >();
    
    // Find all localities supporting the component 
    std::vector<hpx::id_type> localities = hpx::find_all_localities(type);

    assert(!localities.empty());
   
    // Asynchronously deploy the component on every locality.
    // FIXME: Sadly, distributing factory doesn't support constructor args
    // yet, so we have to do this by hand.
    std::vector<AsyncLevelDataRegistryClient<T> > clients;
    clients.reserve(localities.size());

    for (std::size_t i = 0; i < localities.size(); ++i)
    {
        using hpx::components::stubs::runtime_support;
        // Create the component, passing any relevant global information.
        clients.emplace_back(
            runtime_support::create_component_async<
                AsyncLevelDataRegistryServer<T>
            >(localities[i]));
    }

    // Send everyone the list of clients.
    std::vector<hpx::future<void> > futures;
    futures.reserve(localities.size());

    for (std::size_t i = 0; i < clients.size(); ++i)
    {
        futures.emplace_back(clients[i].createDispatcherAsync(clients));
    }

    hpx::lcos::when_all(futures).get();

    return clients;
} //}}}

template <typename T>
hpx::future<void> RegisterALDAsync(
    AsyncLevelData<T>* local_data
    )
{ // {{{
    return AsyncLevelDataRegistryDispatcher<T>::insertAsync(local_data);
} // }}}

template <typename T>
void RegisterALDSync(
    AsyncLevelData<T>* local_data
    )
{ // {{{
    AsyncLevelDataRegistryDispatcher<T>::insertAsync(local_data).get();
} // }}}

template <typename T>
hpx::future<void> ApplyALDAsync(
    std::uint32_t proc
  , std::size_t tag
  , hpx::util::function<void(AsyncLevelData<T>&)>&& f
    )
{ // {{{
    return AsyncLevelDataRegistryDispatcher<T>::applyAsync(
        proc, tag, std::move(f)
    );
} // }}}

template <typename T>
void ApplyALDSync(
    std::uint32_t proc
  , std::size_t tag
  , hpx::util::function<void(AsyncLevelData<T>&)>&& f
    )
{ // {{{
    AsyncLevelDataRegistryDispatcher<T>::applySync(
        proc, tag, std::move(f)
    );
} // }}}

#include "NamespaceFooter.H"

#include "NamespaceVar.H"

// Add function-style overloading for the registeration macro using a forwarding
// variadic macro.
#define HPX_CHOMBO_REGISTER_DALDR_DECLARATION(...)                             \
        HPX_CHOMBO_REGISTER_DALDR_DECLARATION_(__VA_ARGS__)                    \
    /**/

#define HPX_CHOMBO_REGISTER_DALDR_DECLARATION_(...)                            \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                             \
        HPX_CHOMBO_REGISTER_DALDR_DECLARATION_,                                \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                      \
    )(__VA_ARGS__))                                                            \
    /**/

// One argument overload, uses the typename as the registeration name.
#define HPX_CHOMBO_REGISTER_DALDR_DECLARATION_1(T)                             \
    HPX_CHOMBO_REGISTER_DALDR_DECLARATION_2(T, T)                              \
    /**/

// Two argument overload, uses a user-specified name for registeration. This
// version must be used if the type T is not alphanumeric (e.g. if it's a
// template, or namespace qualified). 
#define HPX_CHOMBO_REGISTER_DALDR_DECLARATION_2(T, name)                       \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        CH_XD::AsyncLevelDataRegistryServer<T>::createDispatcher_action,       \
        chombo_AsyncLevelDataRegistryServer_createDispatcher_ ## name          \
    );                                                                         \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        CH_XD::AsyncLevelDataRegistryServer<T>::insert_action,                 \
        chombo_AsyncLevelDataRegistryServer_insert_ ## name                    \
    );                                                                         \
    HPX_REGISTER_ACTION_DECLARATION(                                           \
        CH_XD::AsyncLevelDataRegistryServer<T>::apply_action,                  \
        chombo_AsyncLevelDataRegistryServer_apply_ ## name                     \
    );                                                                         \
    /**/

HPX_CHOMBO_REGISTER_DALDR_DECLARATION(CH_XD::FArrayBox, FArrayBox);

#endif // CHOMBO_50E1AEAC_4E1D_4F10_8A0B_F08E6FED219C

