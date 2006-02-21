#include "RecoTracker/TkDetLayers/interface/TIDLayer.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TIDLayer::TIDLayer(vector<const TIDRing*>& rings):
  theRings(rings.begin(),rings.end()) {}


TIDLayer::~TIDLayer(){
  vector<const TIDRing*>::const_iterator i;
  for (i=theRings.begin(); i!=theRings.end(); i++) {
    delete *i;
  }

} 

vector<const GeomDet*> 
TIDLayer::basicComponents() const{
  cout << "temporary dummy implementation of TIDLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TIDLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TIDLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TIDLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TIDLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



