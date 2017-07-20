#include "RecoVertex/VertexTools/interface/PerigeeRefittedTrackState.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"

AlgebraicVector3 PerigeeRefittedTrackState::momentumVector() const
{
  return momentumAtVertex;
}

std::vector< PerigeeRefittedTrackState::RefCountedRefittedTrackState > 
PerigeeRefittedTrackState::components() const
{
  std::vector<RefCountedRefittedTrackState> result; result.reserve(1);
  result.push_back(RefCountedRefittedTrackState( 
  				const_cast<PerigeeRefittedTrackState*>(this)));
  return result;
}

PerigeeRefittedTrackState::RefCountedRefittedTrackState
PerigeeRefittedTrackState::stateWithNewWeight (const double newWeight) const
{
  return RefCountedRefittedTrackState(
  		new PerigeeRefittedTrackState(theState, momentumAtVertex, newWeight) );
}

TrajectoryStateOnSurface
PerigeeRefittedTrackState::trajectoryStateOnSurface(const Surface & surface) const
{
  AnalyticalPropagator thePropagator(&(theState.theState().parameters().magneticField()), anyDirection);
  TrajectoryStateOnSurface tsos = thePropagator.propagate(freeTrajectoryState(), surface);
  return TrajectoryStateOnSurface (weight(), tsos.globalParameters(),
  	tsos.curvilinearError(), surface) ;
} 

TrajectoryStateOnSurface
PerigeeRefittedTrackState::trajectoryStateOnSurface(const Surface & surface,
				const Propagator & propagator) const
{
  std::unique_ptr<Propagator> thePropagator( propagator.clone());
  thePropagator->setPropagationDirection(anyDirection);

  TrajectoryStateOnSurface tsos = thePropagator->propagate(freeTrajectoryState(), surface);
  return TrajectoryStateOnSurface (weight(), tsos.globalParameters(),
  	tsos.curvilinearError(), surface) ;
}

reco::TransientTrack PerigeeRefittedTrackState::transientTrack() const
{
  TransientTrackFromFTSFactory factory;
  return factory.build(freeTrajectoryState());
}
