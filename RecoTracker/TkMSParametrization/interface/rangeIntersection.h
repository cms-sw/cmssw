#ifndef rangeIntersection_H
#define rangeIntersection_H

#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include <algorithm>

/** Utility for computing efficiently the intersection of two 
 *  one-dimantional intervals.
 *
 *  Pre-condition and expected template argument Range interface:
 *  as for function  rangesIntersect.
 *  If the intervals don't intersect an empty interval is returned.
 */

template <class Range>
inline Range rangeIntersection( const Range& a, const Range& b) {

  return Range( std::max(a.first,b.first),
		std::min(a.second,b.second));
}

template <class Range, class Less>
inline Range rangeIntersection( const Range& a, const Range& b, 
				const Less& less) {

  return Range( std::max( a.first, b.first, less),
		std::min( a.second, b.second, less));
}

#endif

