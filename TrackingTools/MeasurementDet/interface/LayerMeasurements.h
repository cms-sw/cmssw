#ifndef LayerMeasurements_H
#define LayerMeasurements_H


#include <vector>
#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"

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

  /// dummy default constructor (obviously you can't use any object created this way), but it can be needed in some cases
  LayerMeasurements() : theDetSystem(0), theData(0) {}

  /// the constructor that most of the people should be using
  LayerMeasurements( const MeasurementDetSystem& detSystem, const MeasurementTrackerEvent &data) :
    theDetSystem(&detSystem), theData(&data) {}

  std::vector<TrajectoryMeasurement>
  measurements( const DetLayer& layer, 
		const TrajectoryStateOnSurface& startingState,
		const Propagator& prop, 
		const MeasurementEstimator& est) const;

  std::vector<TrajectoryMeasurementGroup>
  groupedMeasurements( const DetLayer& layer, 
		       const TrajectoryStateOnSurface& startingState,
		       const Propagator& prop, 
		       const MeasurementEstimator& est) const;


  void addInvalidMeas( std::vector<TrajectoryMeasurement>& measVec,
		       const DetGroup& group,
		       const DetLayer& layer) const;

  MeasurementDetWithData idToDet(const DetId& id) const {
     return theDetSystem->idToDet(id, *theData);
  }

private:

  const MeasurementDetSystem* theDetSystem;
  const MeasurementTrackerEvent* theData;

};

#endif
