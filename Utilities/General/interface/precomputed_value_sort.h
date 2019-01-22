#ifndef precomputed_value_sort_H
#define precomputed_value_sort_H

/** \file precomputed_value_sort.h
 *  Sort using precomputed values.
 *
 *  precomputed_value_sort behaves like std::sort, but pre-computes the
 *  values used in the sorting using an Extractor, so that the computation
 *  is performed only once per element.
 */

#include <vector>
#include <algorithm>
#include <functional>

template<class RandomAccessIterator, class Extractor, class Compare>
void precomputed_value_sort( RandomAccessIterator begin, RandomAccessIterator end,
                             const Extractor& extr, const Compare& comp )
{
  using Value  = typename std::iterator_traits<RandomAccessIterator>::value_type;
  using Scalar = decltype(extr(*begin));

  std::vector<std::pair<RandomAccessIterator,Scalar>> tmpvec;
  tmpvec.reserve(end-begin);

  // tmpvec holds iterators - does not copy the real objects
  for (RandomAccessIterator i=begin; i!=end; i++) tmpvec.emplace_back(i,extr(*i));

  std::sort(tmpvec.begin(), tmpvec.end(),[&comp](auto const& a, auto const& b){
          return comp(a.second, b.second);
  });

  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<Value> tmpcopy(begin,end);
  for (unsigned int i=0; i < tmpvec.size(); i++) {
    *(begin+i) = std::move(tmpcopy[tmpvec[i].first - begin]);
  }
}

template<class RandomAccessIterator, class Extractor>
void precomputed_value_sort( RandomAccessIterator begin, RandomAccessIterator end,
                             const Extractor& extr )
{
  using Scalar = decltype(extr(*begin));
  precomputed_value_sort(begin, end, extr, std::less<Scalar>());
}

#endif
