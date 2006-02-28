#include "RecoTracker/TkDetLayers/interface/CompositeTECPetal.h"

typedef GeometricSearchDet::DetWithState DetWithState;

CompositeTECPetal::CompositeTECPetal(vector<const TECWedge*>& innerWedges,
				     vector<const TECWedge*>& outerWedges) : 
  theInnerWedges(innerWedges.begin(),innerWedges.end()), 
  theOuterWedges(outerWedges.begin(),outerWedges.end())
{
  theWedges.assign(theInnerWedges.begin(),theInnerWedges.end());
  theWedges.insert(theWedges.end(),theOuterWedges.begin(),theOuterWedges.end());
}


CompositeTECPetal::~CompositeTECPetal(){
  vector<const TECWedge*>::const_iterator i;
  for (i=theWedges.begin(); i!=theWedges.end(); i++) {
    // cout << " Deleting rings " << i-theWedges.begin() << endl;
    delete *i;
  }
} 


const BoundSurface&
CompositeTECPetal::surface() const{
  cout << "temporary dummy implementation of CompositeTECPetal::surface()!!" << endl;
  return thePlane;
}



vector<const GeomDet*> 
CompositeTECPetal::basicComponents() const{
  cout << "temporary dummy implementation of CompositeTECPetal::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}

vector<const GeometricSearchDet*> 
CompositeTECPetal::components() const{
  cout << "temporary dummy implementation of CompositeTECPetal::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
}

  
pair<bool, TrajectoryStateOnSurface>
CompositeTECPetal::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of CompositeTECPetal::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetGroup> 
CompositeTECPetal::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			     const Propagator& prop,
			     const MeasurementEstimator& est) const{

  return vector<DetGroup>();
}



