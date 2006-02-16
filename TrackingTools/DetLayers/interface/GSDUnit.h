#ifndef DetLayers_GSDUnit_h
#define DetLayers_GSDUnit_h

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

#include <vector>

using namespace std;

class GSDUnit : public GeometricSearchDet {

 public:
  GSDUnit() : thePlane(0) {}  
  virtual ~GSDUnit() {};

  /// Returns basic components, if any
  //virtual vector< const GSDUnit*> basicComponents();

  //--- Extension of interface
  virtual const BoundPlane&   specificSurface() const { return *thePlane;}

 private:
  ReferenceCountingPointer<BoundPlane>  thePlane;

};

#endif
