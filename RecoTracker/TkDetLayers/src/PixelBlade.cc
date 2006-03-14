#include "RecoTracker/TkDetLayers/interface/PixelBlade.h"
#include "RecoTracker/TkDetLayers/interface/BladeShapeBuilderFromDet.h"

typedef GeometricSearchDet::DetWithState DetWithState;

PixelBlade::PixelBlade(vector<const GeomDet*>& frontDets,
		       vector<const GeomDet*>& backDets):		       
  theFrontDets(frontDets), theBackDets(backDets) 
{
  theDets.assign(theFrontDets.begin(),theFrontDets.end());
  theDets.insert(theDets.end(),theBackDets.begin(),theBackDets.end());

  theDiskSector      = BladeShapeBuilderFromDet()(theDets);  
  theFrontDiskSector = BladeShapeBuilderFromDet()(theFrontDets);
  theBackDiskSector  = BladeShapeBuilderFromDet()(theBackDets);   

  /*--------- DEBUG INFO --------------
  cout << "DEBUG INFO for PixelBlade" << endl;
  cout << "this: " << this << endl;
  cout << "PixelForwardLayer.surfcace.z(): " 
       << this->surface().position().z() << endl;
  cout << "PixelForwardLayer.surfcace.innerR(): " 
       << this->specificSurface().innerRadius() << endl;
  cout << "PixelForwardLayer.surfcace.outerR(): " 
       << this->specificSurface().outerRadius() << endl;
  -----------------------------------*/

}


vector<const GeometricSearchDet*> 
PixelBlade::components() const{
  return vector<const GeometricSearchDet*>();
}

pair<bool, TrajectoryStateOnSurface>
PixelBlade::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
			const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of PixelBlade::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
PixelBlade::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
PixelBlade::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
				   const Propagator& prop,
				   const MeasurementEstimator& est) const
{
  cout << "temporary dummy implementation of PixelBlade::groupedCompatibleDets()!!" << endl;
  return vector<DetGroup>();
}
