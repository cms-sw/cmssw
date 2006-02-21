#ifndef TkDetLayers_TECPetal_h
#define TkDetLayers_TECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "RecoTracker/TkDetLayers/interface/TECWedge.h"

/** A concrete implementation for TEC petals
 */

class TECPetal : public GeometricSearchDet{
 public:
  TECPetal(vector<const TECWedge*>& innerWedges,
	   vector<const TECWedge*>& outerWedges);

  ~TECPetal();
  
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
  vector<const TECWedge*> theWedges;
  vector<const TECWedge*> theInnerWedges;
  vector<const TECWedge*> theOuterWedges;

  
};


#endif 
