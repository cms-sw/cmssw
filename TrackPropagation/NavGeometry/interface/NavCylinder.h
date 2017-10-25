#ifndef NavCylinder_H
#define NavCylinder_H

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "TrackPropagation/NavGeometry/interface/LinearSearchNavSurfaceImpl.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include <vector>
class Bounds;

class NavCylinder final : public NavSurface {
public:

/*     NavCylinder( const PositionType& pos, const RotationType& rot, Scalar radius) : */
/* 	Surface( pos, rot), Cylinder( pos, rot, radius), NavSurface( pos, rot) {} */

    NavCylinder( const Cylinder* cylinder) : theSurfaceP(cylinder) {}

    // FIXME: restore covariant return type when gcc version upgraded
    //virtual const Cylinder& surface() const {return *theSurfaceP;} 
    const Surface& surface() const override {return *theSurfaceP;} 

    const NavVolume* nextVolume( const NavSurface::LocalPoint& point, 
					 SurfaceOrientation::Side side) const override {
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

  ConstReferenceCountingPointer<Cylinder> theSurfaceP;
  LinearSearchNavSurfaceImpl theImpl;

};

#endif
