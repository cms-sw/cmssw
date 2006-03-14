#ifndef TkDetLayers_TIDLayer_h
#define TkDetLayers_TIDLayer_h


#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIDRing.h"


/** A concrete implementation for TID layer 
 *  built out of TIDRings
 */

class TIDLayer : public RingedForwardLayer{
 public:
  TIDLayer(vector<const TIDRing*>& rings);
  ~TIDLayer();
  
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


  virtual bool hasGroups() const {return true;}

  // DetLayer interface
  virtual Module   module()   const { return silicon;}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  virtual BoundDisk* computeDisk( const vector<const TIDRing*>& rings) const;

  virtual vector<int> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
						     const Propagator& prop ) const;

 protected:  
  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  int findClosest( const vector<GlobalPoint>& ) const;
  
  int findNextIndex( const vector<GlobalPoint>& , int ) const;
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const;
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
  vector<DetGroup> orderAndMergeLevels(const TrajectoryStateOnSurface& tsos,
				       const Propagator& prop,
				       const vector<vector<DetGroup> > groups,
				       const vector<int> indices ) const;




 protected:
  vector<const TIDRing*> theRings;  

  
};


#endif 
