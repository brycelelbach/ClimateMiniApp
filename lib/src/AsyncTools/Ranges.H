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

#if !defined(CHOMBO_2C3E26CF_46EB_47E4_A329_956DEC0B8C09)
#define CHOMBO_2C3E26CF_46EB_47E4_A329_956DEC0B8C09

#include <utility>

#include "NamespaceHeader.H"

template <typename Iterator>
struct iterator_pair_range : std::pair<Iterator, Iterator>
{
    iterator_pair_range(std::pair<Iterator, Iterator>&& x)
      : std::pair<Iterator, Iterator>(std::move(x))
    {}

    iterator_pair_range(std::pair<Iterator, Iterator> const& x)
      : std::pair<Iterator, Iterator>(x)
    {}

    iterator_pair_range(Iterator&& begin, Iterator&& end)
      : std::pair<Iterator, Iterator>(std::move(begin), std::move(end))
    {}

    iterator_pair_range(Iterator const& begin, Iterator const& end)
      : std::pair<Iterator, Iterator>(begin, end)
    {}

    Iterator begin() const
    {
        return this->first;
    }

    Iterator end() const
    {
        return this->second;
    }
};

/// \brief Construct an object which can be iterated with C++11 range-based for
/// loops. Takes an std::pair of InputIterators as arguments.
///
/// Example use case:
///
///    std::multimap<char, int> m;
///    // ...
///    for (auto& key_val : asRange(m.equal_range('c')))
///    {
///        std::cout << key_val.second << "\n";
///    }
template <typename Iterator>
iterator_pair_range<Iterator> asRange(
    std::pair<Iterator, Iterator>&& x
    )
{
    return iterator_pair_range<Iterator>(std::move(x));
}
template <typename Iterator>
iterator_pair_range<Iterator> asRange(
    std::pair<Iterator, Iterator> const& x
    )
{
    return iterator_pair_range<Iterator>(x);
}

/// \brief Construct an object which can be iterated with C++11 range-based for
/// loops. Takes two InputIterators as arguments.
template <typename Iterator>
iterator_pair_range<Iterator> asRange(
    Iterator&& begin
  , Iterator&& end 
    )
{
    return iterator_pair_range<Iterator>(
        std::forward<Iterator>(begin)
      , std::forward<Iterator>(end)
    );
}

#include "NamespaceFooter.H"
#endif // CHOMBO_2C3E26CF_46EB_47E4_A329_956DEC0B8C09
