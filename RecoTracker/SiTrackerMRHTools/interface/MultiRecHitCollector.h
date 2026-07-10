#ifndef SiTrackerMRHTools_MultiRecHitCollector_h
#define SiTrackerMRHTools_MultiRecHitCollector_h

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFwd.h"
#include <vector>

class TrajectoryMeasurement;

class MultiRecHitCollector {
public:
  MultiRecHitCollector(const MeasurementTracker* meas) : theMeasurementTracker(meas) {}
  virtual ~MultiRecHitCollector() = default;
  virtual std::vector<TrajectoryMeasurement> recHits(const Trajectory&,
                                                     const MeasurementTrackerEvent* theMTE) const = 0;

  const MeasurementTracker* getMeasurementTracker() const { return theMeasurementTracker; }

private:
  const MeasurementTracker* theMeasurementTracker;
};

#endif
