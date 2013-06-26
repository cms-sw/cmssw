#include "TrackingTools/DetLayers/interface/ForwardDetRing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
//#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"

#include <algorithm>
#include <cmath>

using namespace std;

ForwardDetRing::~ForwardDetRing(){}

void
ForwardDetRing::compatibleDetsV( const TrajectoryStateOnSurface&,
				 const Propagator&, 
				 const MeasurementEstimator&,
				 std::vector<DetWithState>&) const{
  edm::LogError("DetLayers") << "At the moment not a real implementation" ;  
}

