#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"


pair<bool, TrajectoryStateOnSurface>  
GeomDetCompatibilityChecker::isCompatible(const GeomDet* theDet,
	     const TrajectoryStateOnSurface& tsos,
	     const Propagator& prop, 
	     const MeasurementEstimator& est) const{

  TrajectoryStateOnSurface propSt = prop.propagate( tsos, theDet->specificSurface());
  if ( propSt.isValid()) {
    if ( est.estimate( propSt, theDet->specificSurface()) != 0) {
      return make_pair( true, propSt);
    }
  }
  return make_pair( false, propSt);   
}
 
