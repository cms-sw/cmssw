#ifndef LayerMeasurements_H
#define LayerMeasurements_H


#include <vector>

class GeometricSearchDet;
class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class MeasurementDetSystem;

class LayerMeasurements {
public:

  LayerMeasurements( const MeasurementDetSystem* detSysytem) :
    theDetSystem (detSysytem) {}

  std::vector<TrajectoryMeasurement>
  measurements( const GeometricSearchDet& layer, 
		const TrajectoryStateOnSurface& startingState,
		const Propagator& prop, 
		const MeasurementEstimator& est) const;
  
private:

  const MeasurementDetSystem* theDetSystem;

};

#endif
