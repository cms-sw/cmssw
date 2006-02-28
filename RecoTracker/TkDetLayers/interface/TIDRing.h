#ifndef TkDetLayers_TIDRing_h
#define TkDetLayers_TIDRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"


/** A concrete implementation for TID rings 
 */

class TIDRing : public GeometricSearchDet{
 public:
  TIDRing(vector<const GeomDet*>& innerDets,
	  vector<const GeomDet*>& outerDets);
  ~TIDRing();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return thePlane;}
  
  virtual vector<const GeomDet*> basicComponents() const;
  
  virtual vector<const GeometricSearchDet*> components() const {
    return vector<const GeometricSearchDet*>();}
    

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
  BoundPlane& thePlane; //temporary solution

  
  };


#endif 
