#include "TrackPropagation/NavGeometry/interface/NavCone.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"

TrajectoryStateOnSurface 
NavCone::propagate( const Propagator& prop, 
		    const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagate( startingState, *theSurfaceP); 
}

std::pair<TrajectoryStateOnSurface,double>
NavCone::propagateWithPath( const Propagator& prop, 
		    const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagateWithPath( startingState, *theSurfaceP); 
}

class NavConeNotImplementedDistanceAlongLine : public std::exception {
public:
    NavConeNotImplementedDistanceAlongLine() throw() {}
    ~NavConeNotImplementedDistanceAlongLine() throw() override {}
};

std::pair<bool,double> 
NavCone::distanceAlongLine( const NavSurface::GlobalPoint& pos, 
			    const NavSurface::GlobalVector& dir) const
{
    throw NavConeNotImplementedDistanceAlongLine();
    return std::pair<bool,double>(false,0);
}
