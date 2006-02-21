#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TOBLayer::TOBLayer(vector<const TOBRod*>& innerRods,
		   vector<const TOBRod*>& outerRods) : 
  theInnerRods(innerRods.begin(),innerRods.end()), 
  theOuterRods(outerRods.begin(),outerRods.end())
{
  theRods.assign(theInnerRods.begin(),theInnerRods.end());
  theRods.insert(theRods.end(),theOuterRods.begin(),theOuterRods.end());
}


TOBLayer::~TOBLayer(){
  vector<const TOBRod*>::const_iterator i;
  for (i=theRods.begin(); i!=theRods.end(); i++) {
    delete *i;
  }
} 

vector<const GeomDet*> 
TOBLayer::basicComponents() const{
  cout << "temporary dummy implementation of TOBLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TOBLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TOBLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TOBLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TOBLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



