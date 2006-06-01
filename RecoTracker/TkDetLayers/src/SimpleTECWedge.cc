#include "RecoTracker/TkDetLayers/interface/SimpleTECWedge.h"
#include "RecoTracker/TkDetLayers/interface/ForwardDiskSectorBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

SimpleTECWedge::SimpleTECWedge(const GeomDet* theInputDet):
  theDet(theInputDet)
{
  theDets.push_back(theDet);
  theDiskSector = ForwardDiskSectorBuilderFromDet()( theDets );
}

SimpleTECWedge::~SimpleTECWedge(){

} 


const vector<const GeometricSearchDet*>& 
SimpleTECWedge::components() const{
  throw DetLayerException("SimpleTECWedge doesn't have GeometricSearchDet components");
}

  
pair<bool, TrajectoryStateOnSurface>
SimpleTECWedge::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
			    const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of SimpleTECWedge::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}



vector<DetGroup> 
SimpleTECWedge::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



