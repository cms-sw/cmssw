#ifndef precomputed_value_sort_H
#define precomputed_value_sort_H

/** \file precomputed_value_sort.h
 *  Sort using precomputed values.
 *
 *  precomputed_value_sort behaves like std::sort, but pre-computes the 
 *  values used in the sorting using an Extractor, so that the computation
 *  is performed only once per element.
 *
 *  $Date: 2005/09/21 10:18:42 $
 *  $Revision: 1.1 $
 */

#include <utility>
#include <vector>
#include <algorithm>

namespace {
  template<class T, class Scalar>
  struct LessPair {
    typedef std::pair<T,Scalar> SortPair;
    bool operator()( const SortPair& a, const SortPair& b) {
      return a.second < b.second;
    }
  };

  template <class T, class Scalar, class BinaryPredicate>
  struct ComparePair {
    ComparePair( const BinaryPredicate& cmp) : cmp_(cmp) {}
    typedef std::pair<T,Scalar> SortPair;
    bool operator()( const SortPair& a, const SortPair& b) {
      return cmp_(a.second, b.second);
    }
    BinaryPredicate cmp_;
  };
}


template<class RandomAccessIterator, class Extractor>
void precomputed_value_sort(RandomAccessIterator begin,
			    RandomAccessIterator end,
			    const Extractor& extr) {

  typedef typename Extractor::result_type        Scalar;
  typedef std::pair<RandomAccessIterator,Scalar> SortPair;

  std::vector<SortPair> tmpvec; 
  tmpvec.reserve(end-begin);

  // tmpvec holds iterators - does not copy the real objects
  for (RandomAccessIterator i=begin; i!=end; i++) {
    tmpvec.push_back(SortPair(i,extr(*i)));
  }
  
  std::sort(tmpvec.begin(), tmpvec.end(),
	    LessPair<RandomAccessIterator,Scalar>());    

  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type> tmpcopy(begin,end);
  for (unsigned int i=0; i<tmpvec.size(); i++) {
    *(begin+i) = tmpcopy[tmpvec[i].first - begin];
  }
}


/// Sort using a BinaryPredicate

template<class RandomAccessIterator, class Extractor, class BinaryPredicate>
void precomputed_value_sort( RandomAccessIterator begin,
			     RandomAccessIterator end,
			     const Extractor& extr,
			     const BinaryPredicate& pred) {

  typedef typename Extractor::result_type        Scalar;
  typedef std::pair<RandomAccessIterator,Scalar> SortPair;

  std::vector<SortPair> tmpvec; 
  tmpvec.reserve(end-begin);

  // tmpvec holds iterators - does not copy the real objects
  for (RandomAccessIterator i=begin; i!=end; i++) {
    tmpvec.push_back(SortPair(i,extr(*i)));
  }
  
  std::sort(tmpvec.begin(), tmpvec.end(),
	    ComparePair< RandomAccessIterator,Scalar,BinaryPredicate>(pred));

  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type> tmpcopy(begin,end);
  for (unsigned int i=0; i<tmpvec.size(); i++) {
    *(begin+i) = tmpcopy[tmpvec[i].first - begin];
  }

}

#endif
