#ifndef VolumeCrossReturnType_H
#define VolumeCrossReturnType_H

#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

//class TrajectoryStateOnSurface;

class VolumeCrossReturnType  {

 public:

  typedef ConstReferenceCountingPointer<NavVolume> NavVolumePointer;
  typedef TrajectoryStateOnSurface TSOS;

  VolumeCrossReturnType( const NavVolume* navvol, const TSOS& state, double path) : 
    theNavVolumeP(navvol) , bla(state) , thePathLength(path) { };
  
  // Default constructor needed in /usr/include/c++/3.2.3/bits/stl_map.h:225 --> to use VolumeCrossReturnType in sorted container
  //  VolumeCrossReturnType() :  theNavSurfaceP(0) , theBoundsP(0) , 
  //  theSide(SurfaceOrientation::onSurface) , theGlobalFace(SurfaceOrientation::outer) {};
  //
  VolumeCrossReturnType() {};


  /// Access to actual NavVolume pointer
  const NavVolume * volume() const {return theNavVolumeP ;} 
  
  /// Access to actual Bounds pointer
  const TSOS& tsos() const {return bla ;}
  
  /// Access to actual NavSurface pointer
  double path() const {return thePathLength;} 

  ~VolumeCrossReturnType() {} 


 private:

  const NavVolume*       theNavVolumeP;
  TrajectoryStateOnSurface  bla;
  double                    thePathLength;

};

#endif
