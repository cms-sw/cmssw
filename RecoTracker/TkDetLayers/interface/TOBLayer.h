#ifndef TkDetLayers_TOBLayer_h
#define TkDetLayers_TOBLayer_h


#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "RecoTracker/TkDetLayers/interface/TOBRod.h"

/** A concrete implementation for TOB layer 
 *  built out of TOBRods
 */

class TOBLayer : public RodBarrelLayer{
 public:
  TOBLayer(vector<const TOBRod*>& innerRods,
	   vector<const TOBRod*>& outerRods);
  ~TOBLayer();
  
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

 private:
  vector<const TOBRod*> theRods;
  vector<const TOBRod*> theInnerRods;
  vector<const TOBRod*> theOuterRods;

  
};


#endif 
