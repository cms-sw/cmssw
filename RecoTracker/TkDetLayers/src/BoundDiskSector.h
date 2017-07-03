#ifndef RecoTracker_TkDetLayers_BoundDiskSector_h
#define RecoTracker_TkDetLayers_BoundDiskSector_h
 
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DiskSectorBounds.h"

#pragma GCC visibility push(hidden)
class BoundDiskSector final : public Plane {
 public:
 
 
  ~BoundDiskSector() override {}
 
  BoundDiskSector( const PositionType& pos, 
		   const RotationType& rot, 
		   Bounds* bounds) :
    Plane( pos, rot, bounds) {}
  
  
  float innerRadius() const { return bounds().innerRadius();}
  float outerRadius() const  { return bounds().outerRadius();}
  float phiHalfExtension() const  { return bounds().phiHalfExtension();}

  // hide
  DiskSectorBounds const & bounds() const {
    return static_cast<DiskSectorBounds const &>(Plane::bounds());
  }

};
 
 
#pragma GCC visibility pop
#endif 

