#ifndef DetLayers_DetBelowZ_H
#define DetLayers_DetBelowZ_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include <functional>

/** Predicate that returns true if a Det is at a Z smaller than some value.
 */
typedef GeometricSearchDet Det;

class DetBelowZ : public std::unary_function< const Det*, bool> {
public:
  DetBelowZ( double v) : val(v) {}
  bool operator()( const Det* a) const { return a->position().z() < val;}
private:
  double val;
};

#endif 
