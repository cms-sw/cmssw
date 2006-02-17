#ifndef DetLayers_DetLessR_H
#define DetLayers_DetLessR_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

/** Comparison operator for Dets based on the radius.
 */

typedef GeometricSearchDet Det;

class DetLessR {
public:
  bool operator()( const Det* a, const Det* b) const {

    // multiply by 1+epsilon to make it numericaly stable
    // the epsilon should depend on the scalar precision,
    // this is just a quick fix!
    return a->position().perp()*1.000001 < b->position().perp();
  }
};

#endif 
