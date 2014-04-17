#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"


Propagator::~Propagator() {}

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
Propagator::propagate( TrajectoryStateOnSurface const & state, 
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


FreeTrajectoryState 
Propagator::propagate(const FreeTrajectoryState& ftsStart, 
    const reco::BeamSpot& beamSpot) const{
  throw cms::Exception("Propagator::propagate(FTS,beamSpot) not implemented");
}


std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const TrajectoryStateOnSurface& state, 
			       const Surface& sur) const
{
  assert(0=="Propagator::propagateWithPath generic TSOS not implemented");

}
std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const FreeTrajectoryState& state, 
			       const Surface& sur) const
{

  assert(0=="Propagator::propagateWithPath generic FTS not implemented");

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



std::pair<FreeTrajectoryState, double> 
Propagator::propagateWithPath(const FreeTrajectoryState& ftsStart, 
			      const GlobalPoint& pDest1, const GlobalPoint& pDest2) const{
  throw cms::Exception("Propagator::propagate(FTS,GlobalPoint,GlobalPoint) not implemented");
}
