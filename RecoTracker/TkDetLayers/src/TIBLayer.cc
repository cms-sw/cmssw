#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TIBLayer::TIBLayer(vector<const TIBRing*>& innerRings,
		   vector<const TIBRing*>& outerRings) : 
  theInnerRings(innerRings.begin(),innerRings.end()), 
  theOuterRings(outerRings.begin(),outerRings.end())
{
  theRings.assign(theInnerRings.begin(),theInnerRings.end());
  theRings.insert(theRings.end(),theOuterRings.begin(),theOuterRings.end());
}

TIBLayer::~TIBLayer(){
  vector<const TIBRing*>::const_iterator i;
  for (i=theRings.begin(); i!=theRings.end(); i++) {
    // cout << " Deleting rings " << i-theWedges.begin() << endl;
    delete *i;
  }
} 

vector<const GeomDet*> 
TIBLayer::basicComponents() const{
  cout << "temporary dummy implementation of TIBLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TIBLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		      const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TIBLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TIBLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TIBLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop,
				 const MeasurementEstimator& est) const{
  
  return vector<DetGroup>();
}



