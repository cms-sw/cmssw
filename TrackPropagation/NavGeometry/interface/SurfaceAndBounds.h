#ifndef SurfaceAndBounds_H
#define SurfaceAndBounds_H

#include "TrackPropagation/NavGeometry/interface/NavSurface.h"

class NavSurface;
class Bounds;

class SurfaceAndBounds  {

 public:


  typedef SurfaceOrientation::GlobalFace GlobalFace;
  typedef SurfaceOrientation::Side Side;
  typedef ConstReferenceCountingPointer<NavSurface> NavSurfacePointer;

  SurfaceAndBounds( const NavSurface* navsurf, const Bounds* bounds, Side side, GlobalFace face) : 
    theNavSurfaceP(navsurf) , theBoundsP(bounds) , theSide(side) , theGlobalFace(face) { };
  
  // Default constructor needed in /usr/include/c++/3.2.3/bits/stl_map.h:225 --> to use SurfaceAndBounds in sorted container
  //  SurfaceAndBounds() :  theNavSurfaceP(0) , theBoundsP(0) , 
  //  theSide(SurfaceOrientation::onSurface) , theGlobalFace(SurfaceOrientation::outer) {};
  //
  SurfaceAndBounds() {};


  /// Access to actual NavSurface pointer
  const NavSurface& surface() const {return *theNavSurfaceP ;} 
  
  /// Access to actual Bounds pointer
  const Bounds& bounds() const {return *theBoundsP ;}
  
  /// Access to actual NavSurface pointer
  const SurfaceOrientation::Side side() const {return theSide;} 

  ~SurfaceAndBounds() {} 


 private:

  NavSurfacePointer    theNavSurfaceP;
  const Bounds*        theBoundsP;
  Side                 theSide;
  GlobalFace           theGlobalFace;

};

#endif
