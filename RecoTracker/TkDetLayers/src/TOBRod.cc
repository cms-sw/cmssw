#include "RecoTracker/TkDetLayers/interface/TOBRod.h"
#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TOBRod::TOBRod(vector<const GeomDet*>& innerDets,
	       vector<const GeomDet*>& outerDets):
  theInnerDets(innerDets),theOuterDets(outerDets)
{
  theDets.assign(theInnerDets.begin(),theInnerDets.end());
  theDets.insert(theDets.end(),theOuterDets.begin(),theOuterDets.end());


  RodPlaneBuilderFromDet planeBuilder;
  setPlane( planeBuilder( theDets));
  theInnerPlane = planeBuilder( theInnerDets);
  theOuterPlane = planeBuilder( theOuterDets);

}

TOBRod::~TOBRod(){
  
} 

vector<const GeomDet*> 
TOBRod::basicComponents() const{
  return theDets;
}

vector<const GeometricSearchDet*> 
TOBRod::components() const{
  cout << "temporary dummy implementation of TOBRod::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
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



