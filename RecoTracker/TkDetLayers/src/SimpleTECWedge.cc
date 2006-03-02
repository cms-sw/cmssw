#include "RecoTracker/TkDetLayers/interface/SimpleTECWedge.h"

typedef GeometricSearchDet::DetWithState DetWithState;

SimpleTECWedge::SimpleTECWedge(const GeomDet* theDet){

}

SimpleTECWedge::~SimpleTECWedge(){

} 

/*
const BoundSurface&
SimpleTECWedge::surface() const{
  cout << "temporary dummy implementation of SimpleTECWedge::surface()!!" << endl;
  return thePlane;
}
*/

vector<const GeomDet*> 
SimpleTECWedge::basicComponents() const{
  cout << "temporary dummy implementation of SimpleTECWedge::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}

vector<const GeometricSearchDet*> 
SimpleTECWedge::components() const{
  cout << "temporary dummy implementation of SimpleTECWedge::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
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



