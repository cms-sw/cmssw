#ifndef TkDetLayers_TIBLayer_h
#define TkDetLayers_TIBLayer_h


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBRing.h"

//#include "RecoTracker/TkDetLayers/interface/TkGeometricSearchDet.h"

/** A concrete implementation for TIB layer 
 *  built out of TIBRings
 */

//class TIBLayer : public BarrelDetLayer, public TkGeometricSearchDet{
class TIBLayer : public BarrelDetLayer {
 public:

  TIBLayer(vector<const TIBRing*>& innerRings,
	   vector<const TIBRing*>& outerRings);

  ~TIBLayer();
  
  // GeometricSearchDet interface
  
  virtual vector<const GeomDet*> basicComponents() const;
  
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<GeometricSearchDet::DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& tsos,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const; 
    //{return TkGeometricSearchDet::compatibleDets(tsos,prop,est);}

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return true;};  

  // DetLayer interface
  virtual Module   module()   const { return silicon;}

 private:
  vector<const TIBRing*> theRings;
  vector<const TIBRing*> theInnerRings;
  vector<const TIBRing*> theOuterRings;
  
};


#endif 
