#ifndef DetLayers_DetLessZ_H
#define DetLayers_DetLessZ_H

/** Comparison operator for Dets based on the Z.
 */

#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

inline bool isDetLessZ( const GeometricSearchDet* a, const GeometricSearchDet* b) {

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


#endif 
