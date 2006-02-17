#ifndef DetLayers_DetBelowR_H
#define DetLayers_DetBelowR_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

/** Predicate that returns true if a Det is at a radius smaller than some value.
 */
typedef GeometricSearchDet Det;

class DetBelowR {
public:
  DetBelowR( double v) : val(v) {}
  bool operator()( const Det* a) const { return a->position().perp() < val;}
private:
  double val;
};

#endif 
