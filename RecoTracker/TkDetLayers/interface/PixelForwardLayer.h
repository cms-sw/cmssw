#ifndef TkDetLayers_PixelForwardLayer_h
#define TkDetLayers_PixelForwardLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"


/** A concrete implementation for PixelForward layer 
 *  built out of ForwardPixelBlade
 */

class PixelForwardLayer : public ForwardDetLayer{
 public:
  PixelForwardLayer();
  ~PixelForwardLayer();
  
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


  
};


#endif 
