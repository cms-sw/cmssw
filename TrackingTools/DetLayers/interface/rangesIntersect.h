#ifndef TrackingTools_DetLayers_rangesIntersect_h
#define TrackingTools_DetLayers_rangesIntersect_h

/** Utility for checking efficiently if two one-dimantional intervals
 *  intersect. 
 *  Precondition: the intervals are not empty, i.e. for i in a,b
 *  i.first <= i.second.
 *  The Range template argument is expected to have the std::pair
 *  interface, i.e. for Range instance r r.first is the beginning of
 *  the interval and r.second is the end of the interval.
 */

template <typename Range>
inline bool rangesIntersect( const Range& a, const Range& b) {
  return !( a.first > b.second || b.first > a.second);
}

template <typename Range, typename Less>
inline bool rangesIntersect( const Range& a, const Range& b, 
			     Less const &less) {
  return  !( less(b.second,a.first) || less(a.second,b.first));
}
template <typename Range, typename T>
inline bool rangesIntersect( const Range& a, const Range& b, 
			     bool (*less)(T,T)) {
  return  !( less(b.second,a.first) || less(a.second,b.first));
}
#endif
