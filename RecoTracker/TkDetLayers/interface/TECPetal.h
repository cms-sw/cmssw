#ifndef TkDetLayers_TECPetal_h
#define TkDetLayers_TECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "RecoTracker/TkDetLayers/interface/TECWedge.h"
#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"


/** A concrete implementation for TEC petals
 */

class TECPetal : public GeometricSearchDet{
 public:  
  // GeometricSearchDet interface  
  virtual const BoundSurface& surface() const{return *theDiskSector;}


  virtual std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual bool hasGroups() const {return true;}  
  
  //Extension of the interface
  virtual const BoundDiskSector& specificSurface() const {return *theDiskSector;}

 protected:
  // it needs to be initialized somehow ins the derived class
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;
  
};


#endif 
