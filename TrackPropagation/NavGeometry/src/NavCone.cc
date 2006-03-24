#include "TrackPropagation/NavGeometry/interface/NavCone.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Surface/interface/Bounds.h"

TrajectoryStateOnSurface 
NavCone::propagate( const Propagator& prop, 
		    const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagate( startingState, *theSurfaceP); 
}

class NavConeNotImplementedDistanceAlongLine : public std::exception {
public:
    NavConeNotImplementedDistanceAlongLine() throw() {}
    virtual ~NavConeNotImplementedDistanceAlongLine() throw() {}
};

std::pair<bool,double> 
NavCone::distanceAlongLine( const NavSurface::GlobalPoint& pos, 
			    const NavSurface::GlobalVector& dir) const
{
    throw NavConeNotImplementedDistanceAlongLine();
    return std::pair<bool,double>(false,0);
}
