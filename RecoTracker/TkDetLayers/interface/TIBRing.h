#ifndef TkDetLayers_TIBRing_h
#define TkDetLayers_TIBRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"


/** A concrete implementation for TIB rings 
 */

class TIBRing : public GeometricSearchDet{
 public:
  TIBRing();
  ~TIBRing();
  
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
