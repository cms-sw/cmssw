#ifndef NavSurface_H
#define NavSurface_H

#include "DataFormats/GeometrySurface/interface/Surface.h"

#include <utility>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class NavVolume;
class TrajectoryStateOnSurface;
class Propagator;
class Bounds;

class NavSurface : public ReferenceCountedInConditions // : public virtual Surface 
{
public:

  // NavSurface( const PositionType& pos, const RotationType& rot) : Surface(pos,rot) {}

    typedef Surface::LocalPoint           LocalPoint;
    typedef Surface::LocalVector          LocalVector;
    typedef Surface::GlobalPoint          GlobalPoint;
    typedef Surface::GlobalVector         GlobalVector;
    typedef std::pair<TrajectoryStateOnSurface,double> TSOSwithPath;

    virtual ~NavSurface() {}

    /// Access to actual surface
    virtual const Surface& surface() const = 0; 

    virtual const NavVolume* nextVolume( const LocalPoint& point, 
					 SurfaceOrientation::Side side) const = 0;

/// hook for double dispatch to avoid propagation to generic surface.
    virtual TrajectoryStateOnSurface 
    propagate( const Propagator& prop, const TrajectoryStateOnSurface& startingState) const = 0;

/// hook for double dispatch to avoid propagation to generic surface.
    virtual TSOSwithPath 
    propagateWithPath( const Propagator& prop, const TrajectoryStateOnSurface& startingState) const = 0;

/// Bounds corresponding to a NavVolume if present
    virtual const Bounds* bounds( const NavVolume* vol) = 0;

/// NavVolumes are supposed to call this method to "register" with the NavSurface
// FIXME: should not be public...
    virtual void addVolume( const NavVolume* vol, const Bounds* bounds, 
			    SurfaceOrientation::Side side) = 0;

/// Distance to surface from point pos along line dir.
/// If the half-line does not cross the surface the returned value is (false,undefined),
/// otherwise the returned value is (true,distance)
    virtual std::pair<bool,double> 
    distanceAlongLine( const GlobalPoint& pos, const GlobalVector& dir) const = 0;

    ///Forwarding of part of surface interface for convenience
    LocalPoint  toLocal( const GlobalPoint& p)   const {return surface().toLocal(p);}
    LocalVector toLocal( const GlobalVector& p)  const {return surface().toLocal(p);}
    GlobalPoint  toGlobal( const LocalPoint& p)  const {return surface().toGlobal(p);}
    GlobalVector toGlobal( const LocalVector& p) const {return surface().toGlobal(p);}


};

#endif
