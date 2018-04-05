#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"


Propagator::~Propagator() {}




std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const FreeTrajectoryState& state, 
			       const Surface& sur) const
{
  // try plane first, most probable case (disk "is a" plane too) 
  const Plane* bp = dynamic_cast<const Plane*>(&sur);
  if (bp != nullptr) return propagateWithPath( state, *bp);
  
  // if not plane try cylinder
  const Cylinder* bc = dynamic_cast<const Cylinder*>(&sur);
  if (bc != nullptr) return propagateWithPath( state, *bc);

  // unknown surface - can't do it!
  throw PropagationException("The surface is neither Cylinder nor Plane");
}

std::pair< TrajectoryStateOnSurface, double> 
Propagator::propagateWithPath (const TrajectoryStateOnSurface& state, 
			       const Surface& sur) const
{
  // try plane first, most probable case (disk "is a" plane too) 
  const Plane* bp = dynamic_cast<const Plane*>(&sur);
  if (bp != nullptr) return propagateWithPath( state, *bp);
  
  // if not plane try cylinder
  const Cylinder* bc = dynamic_cast<const Cylinder*>(&sur);
  if (bc != nullptr) return propagateWithPath( state, *bc);

  // unknown surface - can't do it!
  throw PropagationException("The surface is neither Cylinder nor Plane");
}


std::pair<FreeTrajectoryState, double> 
Propagator::propagateWithPath(const FreeTrajectoryState&, 
			      const GlobalPoint&) const{
  throw cms::Exception("Propagator::propagate(FTS,GlobalPoint) not implemented");
  return std::pair<FreeTrajectoryState, double> ();
}
std::pair<FreeTrajectoryState, double> 
Propagator::propagateWithPath(const FreeTrajectoryState&, 
			      const GlobalPoint&, const GlobalPoint&) const{
  throw cms::Exception("Propagator::propagate(FTS,GlobalPoint,GlobalPoint) not implemented");
  return std::pair<FreeTrajectoryState, double> ();

}
std::pair<FreeTrajectoryState, double> 
Propagator::propagateWithPath(const FreeTrajectoryState& ftsStart,  const reco::BeamSpot& beamSpot) const{
  throw cms::Exception("Propagator::propagate(FTS,beamSpot) not implemented");
  return std::pair<FreeTrajectoryState, double> ();
}

