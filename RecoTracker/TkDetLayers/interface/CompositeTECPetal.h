#ifndef TkDetLayers_CompositeTECPetal_h
#define TkDetLayers_CompositeTECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "RecoTracker/TkDetLayers/interface/TECPetal.h"
#include "RecoTracker/TkDetLayers/interface/TECWedge.h"

/** A concrete implementation for TEC petals
 */

class CompositeTECPetal : public TECPetal{
 public:
  CompositeTECPetal(vector<const TECWedge*>& innerWedges,
		    vector<const TECWedge*>& outerWedges);

  ~CompositeTECPetal();
  
  // GeometricSearchDet interface

  virtual const BoundSurface& surface() const;
  
  virtual vector<const GeomDet*> basicComponents() const;

  virtual vector<const GeometricSearchDet*> components() const;
  
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;

 private:
  vector<const TECWedge*> theWedges;
  vector<const TECWedge*> theInnerWedges;
  vector<const TECWedge*> theOuterWedges;

 private:
  BoundPlane& thePlane; //temporary solution
  
};


#endif 
