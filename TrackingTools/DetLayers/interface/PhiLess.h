#ifndef DetLayers_PhiLess_h
#define DetLayers_PhiLess_h

#include <functional>
#include <cmath>

/** Definition of ordering of azimuthal angles.
 *  phi1 is less than phi2 if the angle covered by a point going from
 *  phi1 to phi2 in the counterclockwise direction is smaller than pi.
 */

class PhiLess : public std::binary_function< float, float, bool> {
public:
  bool operator()( float a, float b) const {
    const float pi = 3.141592653592;
    float diff = fmod(b - a, 2*pi);
    if ( diff < 0) diff += 2*pi;
    return diff < pi;
  }
};

#endif 
