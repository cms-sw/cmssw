#ifndef AmplitudeSegmentFP420_h
#define AmplitudeSegmentFP420_h


//#include "G4StepPoint.hh"


#include<vector>
#include "G4ThreeVector.hh"

class AmplitudeSegmentFP420 {
 public:
  AmplitudeSegmentFP420() : _pos(0,0,0), _sigma(0), _amplitude(0) {}
  
  AmplitudeSegmentFP420( float x, float y, float z, float s,float a=1.0) : 
    _pos(x,y,z), _sigma(s), _amplitude(a) {}
  
  const G4ThreeVector& position() const { return _pos;}
  float x()         const { return _pos.x();}
  float y()         const { return _pos.y();}
  float z()         const { return _pos.z();}
  float sigma()     const { return _sigma;}
  float amplitude() const { return _amplitude;}
  AmplitudeSegmentFP420& set_amplitude( float amp) { _amplitude = amp; return *this;} 

 private:

  G4ThreeVector      _pos;
  float           _sigma;
  float           _amplitude;
  };
#endif
