#ifndef DetLayers_PhiLess_h
#define DetLayers_PhiLess_h

#include <functional>
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

/** Definition of ordering of azimuthal angles.
 *  phi1 is less than phi2 if the angle covered by a point going from
 *  phi1 to phi2 in the counterclockwise direction is smaller than pi.
 */

class PhiLess : public std::binary_function< float, float, bool> {
public:
  bool operator()( float a, float b) const {
    return Geom::phiLess(a,b);
  }
};

#endif 
