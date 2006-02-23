#ifndef TkDetLayers_TOBRod_h
#define TkDetLayers_TOBRod_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"


/** A concrete implementation for TOB Rod 
 *  
 */

class TOBRod : public DetRod{
 public:
  TOBRod(vector<const GeomDet*>& innerDets,
	 vector<const GeomDet*>& outerDets);
  ~TOBRod();
  
  // GeometricSearchDet interface
  
  virtual vector<const GeomDet*> basicComponents() const;

  virtual vector<const GeometricSearchDet*> components() const;

  
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


  virtual bool hasGroups() const {return true;}  

 private:
  vector<const GeomDet*> theDets;
  vector<const GeomDet*> theInnerDets;
  vector<const GeomDet*> theOuterDets;

  ReferenceCountingPointer<BoundPlane> theInnerPlane;
  ReferenceCountingPointer<BoundPlane> theOuterPlane;
};


#endif 
