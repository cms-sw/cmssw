#ifndef LayerMeasurements_H
#define LayerMeasurements_H


#include <vector>

class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class TrajectoryMeasurementGroup;
class MeasurementDetSystem;
class DetLayer;
class DetGroup;

class LayerMeasurements {
public:

  LayerMeasurements( const MeasurementDetSystem* detSysytem) :
    theDetSystem (detSysytem) {}

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
  
private:

  const MeasurementDetSystem* theDetSystem;

};

#endif
