#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 

using namespace std;

pair<bool, TrajectoryStateOnSurface>  
GeomDetCompatibilityChecker::isCompatible(const GeomDet* theDet,
					  const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop, 
					  const MeasurementEstimator& est) {
  TrajectoryStateOnSurface propSt = prop.propagate( tsos, theDet->specificSurface());
  if ( propSt.isValid()) {
    if ( est.estimate( propSt, theDet->specificSurface()) ) {
      return make_pair( true, propSt);
    }
  }
  return make_pair( false, propSt);   
}
 
