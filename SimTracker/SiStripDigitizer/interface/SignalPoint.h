#ifndef Tracker_SignalPoint_H
#define Tracker_SignalPoint_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

/**
 * An elementar charge point, with position, sigma from diffusion and Amplitude.
 * That describes the drifted charge seen on the surface of the sensors.
 */
class SignalPoint {
 public:
  SignalPoint() : _pos(0,0), _sigma(0), _amplitude(0) {}
  
  SignalPoint( float x, float y, float s,float a=1.0) : 
    _pos(x,y), _sigma(s), _amplitude(a) {}
  
  const LocalPoint& position() const { return _pos;}
  float x()         const { return _pos.x();}
  float y()         const { return _pos.y();}
  float sigma()     const { return _sigma;}
  float amplitude() const { return _amplitude;}
  SignalPoint& set_amplitude( float amp) { _amplitude = amp; return *this;} 
 private:
  LocalPoint      _pos;
  float           _sigma;
  float           _amplitude;
  };
#endif
