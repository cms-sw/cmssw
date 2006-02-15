#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"

TrajectoryStateOnSurface 
Propagator::propagate( const FreeTrajectoryState& state, 
		       const Surface& sur) const
{
  // try plane first, most probable case (disk "is a" plane too) 
  const Plane* bp = dynamic_cast<const Plane*>(&sur);
  if (bp != 0) return propagate( state, *bp);
  
  // if not plane try cylinder
  const Cylinder* bc = dynamic_cast<const Cylinder*>(&sur);
  if (bc != 0) return propagate( state, *bc);

  // unknown surface - can't do it!
  throw PropagationException("The surface is neither Cylinder nor Plane");
}

TrajectoryStateOnSurface 
Propagator::propagate (const TrajectoryStateOnSurface& state, 
		       const Surface& sur) const
{
  // exactly same code as for FreeTrajectoryState

  // try plane first, most probable case (disk "is a" plane too) 
  const Plane* bp = dynamic_cast<const Plane*>(&sur);
  if (bp != 0) return propagate( state, *bp);
  
  // if not plane try cylinder
  const Cylinder* bc = dynamic_cast<const Cylinder*>(&sur);
  if (bc != 0) return propagate( state, *bc);

  // unknown surface - can't do it!
  throw PropagationException("The surface is neither Cylinder nor Plane");
}

// default impl. avoids the need to redefinition in concrete
// propagators that don't benefit from TSOS vs. FTS
TrajectoryStateOnSurface 
Propagator::propagate (const TrajectoryStateOnSurface& tsos, 
		       const Plane& sur) const
{
  return propagate( *tsos.freeState(), sur);
}

// default impl. avoids the need to redefinition in concrete
// propagators that don't benefit from TSOS vs. FTS
TrajectoryStateOnSurface 
Propagator::propagate (const TrajectoryStateOnSurface& tsos, 
		       const Cylinder& sur) const
{
  return propagate( *tsos.freeState(), sur);
}


std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const FreeTrajectoryState& state, 
			       const Surface& sur) const
{
  // same code as above, only method name changes

  // try plane first, most probable case (disk "is a" plane too) 
  const Plane* bp = dynamic_cast<const Plane*>(&sur);
  if (bp != 0) return propagateWithPath( state, *bp);
  
  // if not plane try cylinder
  const Cylinder* bc = dynamic_cast<const Cylinder*>(&sur);
  if (bc != 0) return propagateWithPath( state, *bc);

  // unknown surface - can't do it!
  throw PropagationException("The surface is neither Cylinder nor Plane");
}

std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const TrajectoryStateOnSurface& state, 
			       const Surface& sur) const
{
  // same code as above, only method name changes

  // try plane first, most probable case (disk "is a" plane too) 
  const Plane* bp = dynamic_cast<const Plane*>(&sur);
  if (bp != 0) return propagateWithPath( state, *bp);
  
  // if not plane try cylinder
  const Cylinder* bc = dynamic_cast<const Cylinder*>(&sur);
  if (bc != 0) return propagateWithPath( state, *bc);

  // unknown surface - can't do it!
  throw PropagationException("The surface is neither Cylinder nor Plane");
}

// default impl. avoids the need to redefinition in concrete
// propagators that don't benefit from TSOS vs. FTS
std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const TrajectoryStateOnSurface& tsos, 
			       const Plane& sur) const
{
  return propagateWithPath( *tsos.freeState(), sur);
}

// default impl. avoids the need to redefinition in concrete
// propagators that don't benefit from TSOS vs. FTS
std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const TrajectoryStateOnSurface& tsos, 
			       const Cylinder& sur) const
{
  return propagateWithPath( *tsos.freeState(), sur);
}
