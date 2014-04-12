#ifndef NavCylinder_H
#define NavCylinder_H

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "TrackPropagation/NavGeometry/interface/NavSurface.h"
#include "TrackPropagation/NavGeometry/interface/LinearSearchNavSurfaceImpl.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include <vector>
class Bounds;

class NavCylinder GCC11_FINAL : public NavSurface {
public:

/*     NavCylinder( const PositionType& pos, const RotationType& rot, Scalar radius) : */
/* 	Surface( pos, rot), Cylinder( pos, rot, radius), NavSurface( pos, rot) {} */

    NavCylinder( const Cylinder* cylinder) : theSurfaceP(cylinder) {}

    // FIXME: restore covariant return type when gcc version upgraded
    //virtual const Cylinder& surface() const {return *theSurfaceP;} 
    virtual const Surface& surface() const {return *theSurfaceP;} 

    virtual const NavVolume* nextVolume( const NavSurface::LocalPoint& point, 
					 SurfaceOrientation::Side side) const {
	return theImpl.nextVolume( point,side);
    }

    virtual TrajectoryStateOnSurface 
    propagate( const Propagator& prop, const TrajectoryStateOnSurface& startingState) const;

    virtual NavSurface::TSOSwithPath 
    propagateWithPath( const Propagator& prop, const TrajectoryStateOnSurface& startingState) const;

    virtual const Bounds* bounds( const NavVolume* vol) { return theImpl.bounds(vol);}

    virtual void addVolume( const NavVolume* vol, const Bounds* bounds, 
			    SurfaceOrientation::Side side) {
	theImpl.addVolume( vol, bounds, side);
    }

    virtual std::pair<bool,double> 
    distanceAlongLine( const NavSurface::GlobalPoint& pos, const NavSurface::GlobalVector& dir) const;

private:

  ConstReferenceCountingPointer<Cylinder> theSurfaceP;
  LinearSearchNavSurfaceImpl theImpl;

};

#endif
