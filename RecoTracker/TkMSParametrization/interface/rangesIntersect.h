#ifndef rangesIntersect_H
#define rangesIntersect_H

/** Utility for checking efficiently if two one-dimantional intervals
 *  intersect. 
 *  Precondition: the intervals are not empty, i.e. for i in a,b
 *  i.first <= i.second.
 *  The Range template argument is expected to have the std::pair
 *  interface, i.e. for Range instance r r.first is the beginning of
 *  the interval and r.second is the end of the interval.
 */

template <class Range>
inline bool rangesIntersect( const Range& a, const Range& b) {
  if ( a.first > b.second || b.first > a.second) return false;
  else return true;
}

template <class Range, class Less>
inline bool rangesIntersect( const Range& a, const Range& b, 
			     const Less& less) {
  if ( less(b.second,a.first) || less(a.second,b.first)) return false;
  else return true;
}
#endif
