#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

DetRod::~DetRod(){}

vector<GeometricSearchDet::DetWithState> 
DetRod::compatibleDets( const TrajectoryStateOnSurface& fts,
			const Propagator& prop, 
			const MeasurementEstimator& est) const
{
  edm::LogError("DetLayers") << "At the moment not a real implementation" ;
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
