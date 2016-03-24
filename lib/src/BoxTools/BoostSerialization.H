#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _CHOMBO_BOOST_SERIALIZATION_H_
#define _CHOMBO_BOOST_SERIALIZATION_H_

#include "IntVect.H"
#include "Interval.H"

#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>

#include "NamespaceHeader.H"

// IntVect

template <typename Archive>
inline void serialize(Archive& ar, IntVect& iv, unsigned)
{
  ar & boost::serialization::make_array(iv.dataPtr(), SpaceDim);
}

// Interval

template <typename Archive>
inline void save(Archive& ar, Interval const& i, unsigned)
{
  int begin = i.begin(), end = i.end();
  ar & begin;
  ar & end;
}

template <typename Archive>
inline void load(Archive& ar, Interval& i, unsigned)
{
  int begin = 0, end = 0;
  ar & begin;
  ar & end;
  i.define(begin, end);
}

template <typename Archive>
inline void serialize(Archive& ar, Interval& i, unsigned version)
{
  boost::serialization::split_free(ar, i, version);
}

// Box

template <typename Archive>
inline void save(Archive& ar, Box const& b, unsigned)
{
  IntVect smallEnd = b.smallEnd(), bigEnd = b.bigEnd(), ixType = b.type();
  ar & smallEnd; 
  ar & bigEnd;
  ar & ixType;
}

template <typename Archive>
inline void load(Archive& ar, Box& b, unsigned)
{
  IntVect smallEnd, bigEnd, ixType;
  ar & smallEnd;
  ar & bigEnd;
  ar & ixType;
  b.define(smallEnd, bigEnd, ixType);
}

template <typename Archive>
inline void serialize(Archive& ar, Box& b, unsigned version)
{
  boost::serialization::split_free(ar, b, version);
}

#include "NamespaceFooter.H"

#endif
