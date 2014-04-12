#include "TrackingTools/DetLayers/interface/DetRod.h"

using namespace std;

DetRod::~DetRod(){}



//obsolete?
/*
float DetRod::zError( const TrajectoryStateOnSurface& tsos,
		      const MeasurementEstimator& est) const{  
  const float nSigmas = 3.f;
  if (tsos.hasError()) {
    return nSigmas * sqrt(tsos.localError().positionError().yy());
  }
  else return nSigmas * 0.5;
}
*/
