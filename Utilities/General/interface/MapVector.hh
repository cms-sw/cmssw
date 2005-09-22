#ifndef STLUTIL_MAPVECTOR_H
#define STLUTIL_MAPVECTOR_H 1

#include <algorithm>
#include <functional>
#include <vector>
#include <utility>

/*
  Author: Jim Kowalkowski
  Date:		8/20/2000

  $Id: MapVector.hh,v 1.4 2000/12/08 22:39:55 jbk Exp $

  This class acts like an std::map, but can be populated like an
  std::vector.  Why? Filling a map with lots of data is time
  consuming.  Vectors fill very fast provided you reserve space.
  The search (find) in this class is binary_search (as good or
  better than std::map).  The trick is that the vector inside
  is sorted by key on first access so that binary_search can
  work.  So this class does not do well for problem that
  require adding an item, then searching for another, adding
  an item, then searching for another, and so on...
	
  There is still a minor problem to resolve here.  The KEY was not
  const, so one could perhaps modify it.  If I made it const
  in value_type, then one cannot sort it or insert items.
  So I made a class called MapVecPair to use instead of std::pair,
  but this means that one can still copy a new "pair" over an existing
  entry (bad thing to do), but one cannot modify the key.
  I still have not figured out a way to prevent copying except
  by priviledged classes (things in MapVector, like its std::vector)
  and prevent reassignment of the key.

*/

template <class KEY, class DATA, class COMP=std::less<KEY>,
	  class Allocator=std::allocator<std::pair<KEY,DATA> > >
class MapVector
{
 public:
  typedef std::pair<KEY,DATA> value_type;
  typedef DATA data_type;
  typedef KEY key_type;
  typedef COMP compare_type;
  typedef Allocator allocator_type;
  typedef std::vector<value_type,allocator_type> collection_type;
  typedef typename collection_type::iterator iterator;
  typedef typename collection_type::const_iterator const_iterator;
  typedef typename collection_type::reference reference;
  typedef typename collection_type::const_reference const_reference;
  typedef typename collection_type::size_type size_type;

  MapVector();

  iterator find(const key_type& key);
  const_iterator find(const key_type& key) const;

  /*
    The choice was made here to give this operator map-like
    behavior instead of vector-like behavior.  This present
    performance problems depending on the type DATA.
    There is also the problem that if the object does not exist,
    it is created.  This causes a search, which could cause a sort
    or resize of the underlying container, which could cause all the
    elements to move position, so the iterators are not valid and
    the references are not likely to be valid.  Be careful about
    holding onto the reference, it is not guarenteed to be correct
    about a search or insert.
  */

  DATA& operator[](const key_type& key);

  // here is the vector-like access to elements
  reference at(size_type s) { return cont.at(s); }
  const_reference at(size_type s) const { return cont.at(s); }

  iterator end() { return cont.end(); }
  const_iterator end() const { return cont.end(); }
  iterator begin() { return cont.begin(); }
  const_iterator begin() const { return cont.begin(); }

  void add(const key_type& key, const data_type& data);
  void push_back(const value_type& newitem);

  iterator insert(iterator i,const value_type& newitem)
  { push_back(newitem); return i;}

  void reserve(typename collection_type::size_type x) { cont.reserve(x); }

  size_type size() const { return cont.size(); }

 private:
  friend class MapVectorTest;
  void sort_() const;
  iterator find_(const key_type& key) const;

  struct comp
  {
    bool operator()(const value_type& a, const value_type& b)
    {
      compare_type c;
      return c(a.first,b.first);
    }
  };

  struct comp2
  {
    bool operator()(const value_type& a, const key_type& b)
    {
      compare_type c;
      return c(a.first,b);
    }
  };

  mutable collection_type cont;
  mutable bool is_sorted_;
};

// -------------------- implementation details -------------------------

template <class KEY,class DATA,class COMP,class Allocator>
inline MapVector<KEY,DATA,COMP,Allocator>::MapVector():is_sorted_(true) { }

template <class KEY,class DATA,class COMP,class Allocator>
inline void MapVector<KEY,DATA,COMP,Allocator>::sort_() const
{
  if(is_sorted_==false)
    {
      std::sort(cont.begin(),cont.end(),comp());
      is_sorted_ = true;
    }
}

template <class KEY,class DATA,class COMP,class Allocator>
inline DATA& MapVector<KEY,DATA,COMP,Allocator>::operator[](const key_type& key)
{
  if (cont.size() == 0) {
    push_back( value_type(key,data_type()) );
    return cont.back().second;
  }

  if (is_sorted_) {

    COMP comp;

    if (comp(cont.back().first, key)) {
      push_back(value_type(key, data_type()));
    }

    if (! comp(key, cont.back().first)) {
      return cont.back().second;
    }
  }

  iterator i = find(key);

  if(i!=end())
    return i->second;
  else
    {
      push_back( value_type(key,data_type()) );
      return cont.back().second;
    }
}

template <class KEY,class DATA,class COMP,class Allocator>
inline typename MapVector<KEY,DATA,COMP,Allocator>::iterator
MapVector<KEY,DATA,COMP,Allocator>::find(const key_type& key)
{
  return find_(key);
}

template <class KEY,class DATA,class COMP,class Allocator>
inline typename MapVector<KEY,DATA,COMP,Allocator>::const_iterator
MapVector<KEY,DATA,COMP,Allocator>::find(const key_type& key) const
{
  return find_(key);
}

template <class KEY,class DATA,class COMP,class Allocator>
inline typename MapVector<KEY,DATA,COMP,Allocator>::iterator
MapVector<KEY,DATA,COMP,Allocator>::find_(const key_type& key) const
{
  sort_();
  iterator x(std::lower_bound(cont.begin(),cont.end(),key,comp2()));
  if(x==cont.end()) return x;
  if(x->first!=key) return cont.end();
  return x;
}

template <class KEY,class DATA,class COMP,class Allocator>
inline void MapVector<KEY,DATA,COMP,Allocator>::push_back(const value_type& newitem)
{
  COMP comp;
  if (cont.size() > 0 && 
      comp(newitem.first, cont.back().first)) is_sorted_ = false;
  cont.push_back(newitem);
}

template <class KEY,class DATA,class COMP,class Allocator>
inline void
MapVector<KEY,DATA,COMP,Allocator>::add(const key_type& key, const data_type& data)
{
  push_back(value_type(key,data));
}

#endif

