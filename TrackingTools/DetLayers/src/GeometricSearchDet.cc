#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Likely.h"

GeometricSearchDet::~GeometricSearchDet() {}

void GeometricSearchDet::compatibleDetsV(const TrajectoryStateOnSurface& startingState,
                                         const Propagator& prop,
                                         const MeasurementEstimator& est,
                                         std::vector<DetWithState>& result) const {
  if UNLIKELY (!hasGroups())
    edm::LogError("DetLayers") << "At the moment not a real implementation";

  // standard implementation of compatibleDets() for class which have
  // groupedCompatibleDets implemented.

  std::vector<DetGroup> vectorGroups;
  groupedCompatibleDetsV(startingState, prop, est, vectorGroups);
  for (auto itDG = vectorGroups.begin(); itDG != vectorGroups.end(); itDG++) {
    for (auto itDGE = itDG->begin(); itDGE != itDG->end(); itDGE++) {
      result.emplace_back(itDGE->det(), itDGE->trajectoryState());
    }
  }
}

void GeometricSearchDet::groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                                                const Propagator&,
                                                const MeasurementEstimator&,
                                                std::vector<DetGroup>&) const {
  edm::LogError("DetLayers") << "At the moment not a real implementation";
}

std::vector<GeometricSearchDet::DetWithState> GeometricSearchDet::compatibleDets(
    const TrajectoryStateOnSurface& startingState, const Propagator& prop, const MeasurementEstimator& est) const {
  std::vector<DetWithState> result;
  compatibleDetsV(startingState, prop, est, result);
  return result;
}

std::vector<DetGroup> GeometricSearchDet::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                                const Propagator& prop,
                                                                const MeasurementEstimator& est) const {
  std::vector<DetGroup> result;
  groupedCompatibleDetsV(startingState, prop, est, result);
  return result;
}
