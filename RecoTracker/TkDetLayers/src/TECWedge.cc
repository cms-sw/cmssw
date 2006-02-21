#include "RecoTracker/TkDetLayers/interface/TECWedge.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TECWedge::TECWedge(){

}

TECWedge::~TECWedge(){

} 

vector<const GeomDet*> 
TECWedge::basicComponents() const{
  cout << "temporary dummy implementation of TECWedge::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TECWedge::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TECWedge::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TECWedge::compatibleDets( const TrajectoryStateOnSurface& startingState,
		      const Propagator& prop, 
		      const MeasurementEstimator& est) const{

  // standard implementation of compatibleDets() for class which have 
  // groupedCompatibleDets implemented.
  // This code should be moved in a common place intead of being 
  // copied many times.
  
  vector<DetWithState> result;  
  vector<DetGroup> vectorGroups = groupedCompatibleDets(startingState,prop,est);
  for(vector<DetGroup>::const_iterator itDG=vectorGroups.begin();
      itDG!=vectorGroups.end();itDG++){
    for(vector<DetGroupElement>::const_iterator itDGE=itDG->begin();
	itDGE!=itDG->end();itDGE++){
      result.push_back(DetWithState(itDGE->det(),itDGE->trajectoryState()));
    }
  }
  return result;  
}


vector<DetGroup> 
TECWedge::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



