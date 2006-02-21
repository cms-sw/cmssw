#include "RecoTracker/TkDetLayers/interface/TOBRod.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TOBRod::TOBRod(){

}
  


TOBRod::~TOBRod(){

} 

vector<const GeomDet*> 
TOBRod::basicComponents() const{
  cout << "temporary dummy implementation of TOBRod::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TOBRod::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TOBRod::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TOBRod::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TOBRod::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



