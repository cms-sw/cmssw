#ifndef TkDetLayers_TECLayer_h
#define TkDetLayers_TECLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/TECPetal.h"

/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

class TECLayer : public ForwardDetLayer{
 public:
  TECLayer(vector<const TECPetal*>& innerPetals,
	   vector<const TECPetal*>& outerPetals);
  ~TECLayer();
  
  // GeometricSearchDet interface
  
  virtual vector<const GeomDet*> basicComponents() const;
  
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return true;;};  

  // DetLayer interface
  virtual Module   module()   const { return silicon;}

 private:
  vector<const TECPetal*> thePetals;
  vector<const TECPetal*> theInnerPetals;
  vector<const TECPetal*> theOuterPetals;

  
};


#endif 
