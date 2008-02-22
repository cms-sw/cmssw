#ifndef RecoTracker_TkDetLayers_BoundDiskSector_h
#define RecoTracker_TkDetLayers_BoundDiskSector_h
 
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

class BoundDiskSector : public BoundPlane {
 public:
 
 
  virtual ~BoundDiskSector() {}
 
  BoundDiskSector( const PositionType& pos, 
		   const RotationType& rot, 
		   Bounds* bounds) : Surface( pos,rot),
    BoundPlane( pos, rot, bounds) {}
  
  BoundDiskSector( const PositionType& pos, 
		   const RotationType& rot, 
		   const Bounds& bounds) : Surface( pos,rot),
    BoundPlane( pos, rot, bounds) {}
  
  float innerRadius() const;
  float outerRadius() const;
  float phiExtension() const;
 
};
 
 
#endif 

