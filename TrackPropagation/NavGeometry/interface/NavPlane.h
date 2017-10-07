#ifndef NavPlane_H
#define NavPlane_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "TrackPropagation/NavGeometry/interface/LinearSearchNavSurfaceImpl.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include <vector>
class Bounds;

class NavPlane final : public NavSurface {
public:

/*     NavPlane( const PositionType& pos, const RotationType& rot) : */
/* 	Surface( pos, rot), Plane( pos, rot), NavSurface( pos, rot) {} */

    typedef ConstReferenceCountingPointer<Plane> PlanePointer;

    // NavPlane( const Plane* plane) : theSurfaceP(plane) {}
  
    NavPlane( PlanePointer plane) : theSurfaceP(plane) {}

    // FIXME: restore covariant return type when gcc version upgraded
    //virtual const Plane& surface() const {return *theSurfaceP;} 
    const Surface& surface() const override {return *theSurfaceP;} 

    const NavVolume* nextVolume( const NavSurface::LocalPoint& point, 
					 SurfaceOrientation::Side side) const override{
	return theImpl.nextVolume( point,side);

    }

    TrajectoryStateOnSurface 
    propagate( const Propagator& prop, const TrajectoryStateOnSurface& startingState) const override;

    NavSurface::TSOSwithPath 
    propagateWithPath( const Propagator& prop, const TrajectoryStateOnSurface& startingState) const override;

    const Bounds* bounds( const NavVolume* vol) override { return theImpl.bounds(vol);}

    void addVolume( const NavVolume* vol, const Bounds* bounds, 
			    SurfaceOrientation::Side side) override {
	theImpl.addVolume( vol, bounds, side);
    }

    std::pair<bool,double> 
    distanceAlongLine( const NavSurface::GlobalPoint& pos, const NavSurface::GlobalVector& dir) const override;

private:

  PlanePointer theSurfaceP;
  LinearSearchNavSurfaceImpl theImpl;

};

#endif
