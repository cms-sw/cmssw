#include "TrackingTools/DetLayers/interface/ForwardDetRing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
//#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"

#include <algorithm>
#include <cmath>

using namespace std;

ForwardDetRing::~ForwardDetRing(){}

vector<ForwardDetRing::DetWithState> 
ForwardDetRing::compatibleDets( const TrajectoryStateOnSurface& fts,
				const Propagator& prop, 
				const MeasurementEstimator& est) const
{
  edm::LogError("DetLayers") << "At the moment not a real implementation" ;
  return vector<ForwardDetRing::DetWithState>();
}

