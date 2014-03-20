#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"


Propagator::~Propagator() {}

TrajectoryStateOnSurface 
Propagator::propagate( const FreeTrajectoryState& state, 
		       const Surface& sur)
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
		       const Surface& sur)
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
		       const Plane& sur)
{
  // Protect against null propagations
  if (fabs(sur.toLocal(tsos.globalPosition()).z())<1e-5) {
    // Still have to tarnsform the r.f.!
    return TrajectoryStateOnSurface(*tsos.freeState(), sur);
  }
  return propagate( *tsos.freeState(), sur);
}

// default impl. avoids the need to redefinition in concrete
// propagators that don't benefit from TSOS vs. FTS
TrajectoryStateOnSurface 
Propagator::propagate (const TrajectoryStateOnSurface& tsos, 
		       const Cylinder& sur)
{
  return propagate( *tsos.freeState(), sur);
}

FreeTrajectoryState 
Propagator::propagate(const FreeTrajectoryState& ftsStart, 
    const reco::BeamSpot& beamSpot){
  throw cms::Exception("Propagator::propagate(FTS,beamSpot) not implemented");
}


std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const FreeTrajectoryState& state, 
			       const Surface& sur)
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
			       const Surface& sur)
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
			       const Plane& sur)
{
  return propagateWithPath( *tsos.freeState(), sur);
}

// default impl. avoids the need to redefinition in concrete
// propagators that don't benefit from TSOS vs. FTS
std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const TrajectoryStateOnSurface& tsos, 
			       const Cylinder& sur)
{
  return propagateWithPath( *tsos.freeState(), sur);
}

std::pair<FreeTrajectoryState, double> 
Propagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
			      const GlobalPoint& pDest1, const GlobalPoint& pDest2){
  throw cms::Exception("Propagator::propagate(FTS,GlobalPoint,GlobalPoint) not implemented");
}
