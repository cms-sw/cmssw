#include "RecoTracker/TkDetLayers/interface/TECLayer.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TECLayer::TECLayer(vector<const TECPetal*>& innerPetals,
		   vector<const TECPetal*>& outerPetals) : 
  theInnerPetals(innerPetals.begin(),innerPetals.end()), 
  theOuterPetals(outerPetals.begin(),outerPetals.end())
{
  thePetals.assign(theInnerPetals.begin(),theInnerPetals.end());
  thePetals.insert(thePetals.end(),theOuterPetals.begin(),theOuterPetals.end());
}



TECLayer::~TECLayer(){
  vector<const TECPetal*>::const_iterator i;
  for (i=thePetals.begin(); i!=thePetals.end(); i++) {
    delete *i;
  }
} 

vector<const GeomDet*> 
TECLayer::basicComponents() const{
  cout << "temporary dummy implementation of TECLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TECLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TECLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TECLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TECLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



