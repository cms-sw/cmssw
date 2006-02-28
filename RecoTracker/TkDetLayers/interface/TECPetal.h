#ifndef TkDetLayers_TECPetal_h
#define TkDetLayers_TECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "RecoTracker/TkDetLayers/interface/TECWedge.h"

/** A concrete implementation for TEC petals
 */

class TECPetal : public GeometricSearchDet{
 public:  
  // GeometricSearchDet interface  
  virtual vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual bool hasGroups() const {return true;}  

 private:

  
};


#endif 
