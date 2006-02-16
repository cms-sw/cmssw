#include "TrackingTools/DetLayers/interface/ForwardDetRing.h"
//#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"
//#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"

#include <algorithm>
#include <cmath>


ForwardDetRing::~ForwardDetRing(){}

vector<ForwardDetRing::DetWithState> 
ForwardDetRing::compatibleDets( const TrajectoryStateOnSurface& fts,
				const Propagator& prop, 
				const MeasurementEstimator& est) const
{
  cout << "At the moment not a real implementation" << endl;
  return vector<ForwardDetRing::DetWithState>();
}

