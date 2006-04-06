#ifndef SurfaceAndBounds_H
#define SurfaceAndBounds_H

#include "Geometry/Surface/interface/Surface.h"

class NavSurface;
class Bounds;

class SurfaceAndBounds  {
 public:
  
  SurfaceAndBounds( const NavSurface* navsurf =0  , const Bounds* bounds =0 , SurfaceOrientation::Side side = SurfaceOrientation::onSurface ) : 
    TheNavSurfaceP(navsurf) , TheBoundsP(bounds) , TheSide(side) { };
  
  //  SurfaceAndBounds( const NavSurface* navsurf =0  , const Bounds* bounds =0, SurfaceOrientation::Side side = SurfaceOrientation::onSurface ) : 
  //  TheNavSurfaceP(navsurf) , TheBoundsP(bounds), TheSide(side) {};
  

  /// Access to actual NavSurface pointer
  const NavSurface& surface() const {return *TheNavSurfaceP ;} 
  
  /// Access to actual Bounds pointer
  const Bounds& bounds() const {return *TheBoundsP ;}
  
  /// Access to actual NavSurface pointer
  const SurfaceOrientation::Side side() const {return TheSide;} 

  ~SurfaceAndBounds() {} 


 private:

    const NavSurface*    TheNavSurfaceP;
    const Bounds*        TheBoundsP;
    SurfaceOrientation::Side  TheSide;

};

#endif
