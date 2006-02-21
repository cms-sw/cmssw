#ifndef TkDetLayers_PixelBarrelLayer_h
#define TkDetLayers_PixelBarrelLayer_h


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelRod.h"


/** A concrete implementation for PixelBarrel layer 
 *  built out of BarrelPixelRod
 */

class PixelBarrelLayer : public BarrelDetLayer{
 public:
  PixelBarrelLayer(vector<const PixelRod*>& innerRods,
		   vector<const PixelRod*>& outerRods);
  
  ~PixelBarrelLayer();
  
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


  virtual bool hasGroups() const {return true;;;}

 private:
  vector<const PixelRod*> theRods;
  vector<const PixelRod*> theInnerRods;
  vector<const PixelRod*> theOuterRods;

  
};


#endif 
