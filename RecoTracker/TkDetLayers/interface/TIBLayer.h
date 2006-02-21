#ifndef TkDetLayers_TIBLayer_h
#define TkDetLayers_TIBLayer_h


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBRing.h"


/** A concrete implementation for TIB layer 
 *  built out of TIBRings
 */

class TIBLayer : public BarrelDetLayer{
 public:

  TIBLayer(vector<const TIBRing*>& innerRings,
	   vector<const TIBRing*>& outerRings);

  ~TIBLayer();
  
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
  vector<const TIBRing*> theRings;
  vector<const TIBRing*> theInnerRings;
  vector<const TIBRing*> theOuterRings;
  
};


#endif 
