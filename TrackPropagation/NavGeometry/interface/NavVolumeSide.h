#ifndef NavVolumeSide_H
#define NavVolumeSide_H

#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

/** Class for delimiding surface of a NavVolume.
 *  The additional information with respect to NavSurface that is needed
 *  to define <BR>
 *  a) which side of the Surface the volume is (enumerator Surface::Side) <BR>
 *  b) which face of the volume this surface represents (enumerator GlobalFace). 
 *     Only 6 possible values for volume face are defined.
 */

class NavVolumeSide {
public:
  typedef SurfaceOrientation::GlobalFace GlobalFace;
  typedef SurfaceOrientation::Side Side;
  
  typedef ReferenceCountingPointer<NavSurface>    SurfacePointer;

  NavVolumeSide( NavSurface* surf, GlobalFace gSide, Side sSide) : 
    theSurface( surf),  theGlobalFace( gSide), theSurfaceSide( sSide) {}

  NavVolumeSide( SurfacePointer surf, GlobalFace gSide, Side sSide) : 
    theSurface( surf),  theGlobalFace( gSide), theSurfaceSide( sSide) {}

  NavSurface& mutableSurface() const {return *theSurface;}

  const NavSurface& surface() const {return *theSurface;}

  GlobalFace globalFace() const { return theGlobalFace;}

  Side  surfaceSide() const {return theSurfaceSide;}

private:

  SurfacePointer theSurface;
  GlobalFace     theGlobalFace;
  Side           theSurfaceSide;

};


#endif
