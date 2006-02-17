#ifndef DetLayers_DetLessPhi_H
#define DetLayers_DetLessPhi_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

/** Comparison operator for Dets based on the phi angle.
 */

typedef GeometricSearchDet Det;

class DetLessPhi {
public:
  bool operator()( const Det* a, const Det* b) const {
    return a->position().phi() < b->position().phi();
  }
};

#endif 
