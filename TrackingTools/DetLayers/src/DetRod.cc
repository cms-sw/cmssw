#include "TrackingTools/DetLayers/interface/DetRod.h"

DetRod::~DetRod(){}

vector<GeometricSearchDet::DetWithState> 
DetRod::compatibleDets( const TrajectoryStateOnSurface& fts,
			const Propagator& prop, 
			const MeasurementEstimator& est) const
{
  cout << "At the moment not a real implementation" << endl;
  return vector<DetWithState>();
}


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
