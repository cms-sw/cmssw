#ifndef LayerMeasurements_H
#define LayerMeasurements_H


#include <vector>

class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class MeasurementDetSystem;
class DetLayer;

class LayerMeasurements {
public:

  LayerMeasurements( const MeasurementDetSystem* detSysytem) :
    theDetSystem (detSysytem) {}

  std::vector<TrajectoryMeasurement>
  measurements( const DetLayer& layer, 
		const TrajectoryStateOnSurface& startingState,
		const Propagator& prop, 
		const MeasurementEstimator& est) const;
  
private:

  const MeasurementDetSystem* theDetSystem;

};

#endif
