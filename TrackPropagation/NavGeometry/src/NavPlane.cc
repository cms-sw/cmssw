#include "TrackPropagation/NavGeometry/interface/NavPlane.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"

TrajectoryStateOnSurface 
NavPlane::propagate( const Propagator& prop, 
		     const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagate( startingState, *theSurfaceP); 
}

std::pair<TrajectoryStateOnSurface,double>
NavPlane::propagateWithPath( const Propagator& prop, 
		     const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagateWithPath( startingState, *theSurfaceP); 
}

std::pair<bool,double> 
NavPlane::distanceAlongLine( const NavSurface::GlobalPoint& pos, 
			     const NavSurface::GlobalVector& dir) const
{
    StraightLinePlaneCrossing pc( pos.basicVector(), dir.basicVector());
    return pc.pathLength(*theSurfaceP);
}
