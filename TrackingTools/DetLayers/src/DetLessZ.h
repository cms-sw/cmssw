#ifndef DetLayers_DetLessZ_H
#define DetLayers_DetLessZ_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

/** Comparison operator for Dets based on the Z.
 */

typedef GeometricSearchDet Det;

class DetLessZ : public std::binary_function<const Det*, const Det*, bool> {
public:
  bool operator()( const Det* a, const Det* b) const {

    // multiply by 1+epsilon to make it numericaly stable
    // the epsilon should depend on the scalar precision,
    // this is just a quick fix!
    if (a->position().z() > 0) {
      return a->position().z()*1.000001 < b->position().z();
    }
    else if (b->position().z() < 0) {
      return a->position().z() < b->position().z()*1.000001;
    }
    else return true;
  }

};

#endif 
