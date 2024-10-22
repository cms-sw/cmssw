#ifndef LayerMeasurements_H
#define LayerMeasurements_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#include <vector>

class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class TrajectoryMeasurementGroup;
class MeasurementTrackerEvent;
class DetLayer;
class DetGroup;

class LayerMeasurements {
public:
  LayerMeasurements(const MeasurementDetSystem& detSystem, const MeasurementTrackerEvent& data)
      : detSystem_(detSystem), data_(data) {}

  // return just valid hits, no sorting (for seeding mostly)
  std::vector<BaseTrackerRecHit*> recHits(const DetLayer& layer,
                                          const TrajectoryStateOnSurface& startingState,
                                          const Propagator& prop,
                                          const MeasurementEstimator& est) const;

  std::vector<TrajectoryMeasurement> measurements(const DetLayer& layer,
                                                  const TrajectoryStateOnSurface& startingState,
                                                  const Propagator& prop,
                                                  const MeasurementEstimator& est) const;

  std::vector<TrajectoryMeasurementGroup> groupedMeasurements(const DetLayer& layer,
                                                              const TrajectoryStateOnSurface& startingState,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est) const;

  MeasurementDetWithData idToDet(const DetId& id) const { return detSystem_.idToDet(id, data_); }

private:
  MeasurementDetSystem const& detSystem_;
  MeasurementTrackerEvent const& data_;
};

#endif
