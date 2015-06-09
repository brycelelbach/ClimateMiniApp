/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_534FF53F_27F5_4E27_8089_7EC8BDD42481)
#define CHOMBO_534FF53F_27F5_4E27_8089_7EC8BDD42481

#include <iostream>
#include <tuple>

#include <boost/mpl/string.hpp>

#include "NamespaceHeader.H"

/// \brief Provides a recursively-defined static method \a print() which 
/// streams the elements of an \a std::tuple. The \a Delimiter template
/// parameter should be a \a boost::mpl::string specifying the delimiter that
/// should be streamed in between elements of the tuple.
///
/// Example use case:
///
///     template <typename... T>
///     std::ostream& operator<<(std::ostream& os, std::tuple<T...> const& t)
///     {
///         typedef boost::mpl::string<' '> del;
///         TuplePrinter<std::tuple<T...>, del, 0, sizeof...(T)-1>::print(os, t);
///         return os;
///     }
template <typename T, typename Delimiter, std::size_t N, std::size_t Last>
struct TuplePrinter
{
    static void print(std::ostream& out, T const& value)
    {
        out << std::get<N>(value) << boost::mpl::c_str<Delimiter>::value;
        TuplePrinter<T, Delimiter, N + 1, Last>::print(out, value);
    }
};

template <typename T, typename Delimiter, std::size_t N>
struct TuplePrinter<T, Delimiter, N, N>
{
    static void print(std::ostream& out, T const& value)
    {
        out << std::get<N>(value);
    }
};

/// \brief A utility class for streaming data in the Comma Separate Value (CSV)
/// format. 
template <typename... T>
struct CSVTuple
{
    typedef std::tuple<T...> TupleType;

    CSVTuple(TupleType const& data) : m_data(data) {}

    CSVTuple(TupleType&& data) : m_data(data) {}

    // Workaround for bug in ICPC 15.0.1
    static std::size_t constexpr Last = sizeof...(T)-1;

    friend std::ostream& operator<<(std::ostream& os, CSVTuple<T...> const& t)
    {
        typedef boost::mpl::string<','> del;
        TuplePrinter<TupleType, del, 0, Last>::print(os, t.m_data);
        return os;
    }

  private:
    TupleType m_data;
};

template <typename T>
struct SpecialDecay
{
    using type = typename std::decay<T>::type;
};
 
template <typename T>
struct SpecialDecay<std::reference_wrapper<T> >
{
    using type = T&;
};

/// \brief Creates a streamable object that will output \a elements in the
/// Comma Separate Value (CSV) format.
template <typename... T>
CSVTuple<typename SpecialDecay<T>::type...> StreamCSV(T&&... elements)
{
    typedef CSVTuple<typename SpecialDecay<T>::type...> type;
    return type(std::make_tuple(std::forward<T>(elements)...));
}

#include "NamespaceFooter.H"
#endif // CHOMBO_534FF53F_27F5_4E27_8089_7EC8BDD42481
