#ifndef RecoTracker_TkDetLayers_BoundDiskSector_h
#define RecoTracker_TkDetLayers_BoundDiskSector_h
 
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "RecoTracker/TkDetLayers/interface/DiskSectorBounds.h"

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
  
  float innerRadius() const { return bound().innerRadius();}
  float outerRadius() const  { return bound().outerRadius();}
  float phiExtension() const  { return bound().phiExtension();}

  // hide
  DiskSectorBounds const & bounds() const {
    return static_cast<DiskSectorBounds const &>(BoundPlane::bounds());
  }

};
 
 
#endif 

