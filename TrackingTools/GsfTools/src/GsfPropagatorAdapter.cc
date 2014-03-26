#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"

#include "TrackingTools/GsfTools/src/MultiStatePropagation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <atomic>

GsfPropagatorAdapter::GsfPropagatorAdapter (const Propagator& aPropagator) :
  Propagator(aPropagator.propagationDirection()),
  thePropagator(aPropagator.clone()) {}

std::pair<TrajectoryStateOnSurface,double>
GsfPropagatorAdapter::propagateWithPath (const TrajectoryStateOnSurface& tsos,
					 const Plane& plane) const {
  MultiStatePropagation<Plane> multiPropagator(*thePropagator);
  return multiPropagator.propagateWithPath(tsos,plane);
}

std::pair<TrajectoryStateOnSurface,double>
GsfPropagatorAdapter::propagateWithPath (const TrajectoryStateOnSurface& tsos,
					 const Cylinder& cylinder) const {
  MultiStatePropagation<Cylinder> multiPropagator(*thePropagator);
  return multiPropagator.propagateWithPath(tsos,cylinder);
}

std::pair<TrajectoryStateOnSurface,double>
GsfPropagatorAdapter::propagateWithPath (const FreeTrajectoryState& fts,
					 const Plane& plane) const {
  /// use counter in MessageLogger?
  static std::atomic<int> nWarn{0};
  if ( nWarn++<5 )
    edm::LogInfo("GsfPropagatorAdapter") << "GsfPropagator used from FTS = single state mode!";
  return thePropagator->propagateWithPath(fts,plane);
}

std::pair<TrajectoryStateOnSurface,double>
GsfPropagatorAdapter::propagateWithPath (const FreeTrajectoryState& fts,
					 const Cylinder& cylinder) const {
  /// use counter in MessageLogger?
  static std::atomic<int> nWarn{0};
  if ( nWarn++<5 )
    edm::LogInfo("GsfPropagatorAdapter") << "GsfPropagator used from FTS = single state mode!";
  return thePropagator->propagateWithPath(fts,cylinder);
}

void GsfPropagatorAdapter::setPropagationDirection (PropagationDirection dir) {
  thePropagator->setPropagationDirection(dir);
  Propagator::setPropagationDirection(dir);
}
