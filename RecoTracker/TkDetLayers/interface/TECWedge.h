#ifndef TkDetLayers_TECWedge_h
#define TkDetLayers_TECWedge_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"


/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

class TECWedge : public GeometricSearchDet{
 public:
    // GeometricSearchDet interface
  virtual vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual bool hasGroups() const {return true;}
  
};


#endif 
