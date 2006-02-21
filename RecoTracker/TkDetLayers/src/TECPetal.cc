#include "RecoTracker/TkDetLayers/interface/TECPetal.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TECPetal::TECPetal(vector<const TECWedge*>& innerWedges,
		   vector<const TECWedge*>& outerWedges) : 
  theInnerWedges(innerWedges.begin(),innerWedges.end()), 
  theOuterWedges(outerWedges.begin(),outerWedges.end())
{
  theWedges.assign(theInnerWedges.begin(),theInnerWedges.end());
  theWedges.insert(theWedges.end(),theOuterWedges.begin(),theOuterWedges.end());
}


TECPetal::~TECPetal(){
  vector<const TECWedge*>::const_iterator i;
  for (i=theWedges.begin(); i!=theWedges.end(); i++) {
    // cout << " Deleting rings " << i-theWedges.begin() << endl;
    delete *i;
  }
} 



vector<const GeomDet*> 
TECPetal::basicComponents() const{
  cout << "temporary dummy implementation of TECPetal::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TECPetal::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TECPetal::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TECPetal::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TECPetal::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



