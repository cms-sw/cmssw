#ifndef TkDetLayers_TIDRing_h
#define TkDetLayers_TIDRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"


/** A concrete implementation for TID rings 
 */

class TIDRing : public GeometricSearchDet{
 public:
  TIDRing();
  ~TIDRing();
  
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
