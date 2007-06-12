#include "TrackPropagation/NavGeometry/interface/NavCylinder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineCylinderCrossing.h"

TrajectoryStateOnSurface 
NavCylinder::propagate( const Propagator& prop, 
		     const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagate( startingState, *theSurfaceP); 
}

std::pair<TrajectoryStateOnSurface,double>
NavCylinder::propagateWithPath( const Propagator& prop, 
		     const TrajectoryStateOnSurface& startingState) const
{
    return prop.propagateWithPath( startingState, *theSurfaceP); 
}

std::pair<bool,double> 
NavCylinder::distanceAlongLine( const NavSurface::GlobalPoint& pos, 
				const NavSurface::GlobalVector& dir) const
{
    StraightLineCylinderCrossing pc( toLocal(pos), toLocal(dir));
    return pc.pathLength(*theSurfaceP);
}
