#ifndef TkDetLayers_PixelForwardLayer_h
#define TkDetLayers_PixelForwardLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBlade.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"


/** A concrete implementation for PixelForward layer 
 *  built out of ForwardPixelBlade
 */

class PixelForwardLayer : public ForwardDetLayer{
 public:
  PixelForwardLayer(vector<const PixelBlade*>& blades);
  ~PixelForwardLayer();
  
  // GeometricSearchDet interface
  
  virtual vector<const GeomDet*> basicComponents() const;

  virtual vector<const GeometricSearchDet*> components() const {return theComps;}
  
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
  virtual Module   module()   const { return pixel;}
  

 private:  
  virtual BoundDisk* computeDisk(const vector<const PixelBlade*>& blades) const;    
  // methods for groupedCompatibleDets implementation
  void computeHelicity();

  struct SubTurbineCrossings {
    SubTurbineCrossings(){};
    SubTurbineCrossings( int ci, int ni, float nd) : 
      closestIndex(ci), nextIndex(ni), nextDistance(nd) {}
    
    int   closestIndex;
    int   nextIndex;
    float nextDistance;
  };
  
  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubTurbineCrossings& crossings,
			float window, 
			vector<DetGroup>& result) const;
  
  SubTurbineCrossings 
    computeCrossings( const TrajectoryStateOnSurface& startingState,
		      PropagationDirection propDir) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
 private:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;
  BinFinderType    theBinFinder;

  vector<const PixelBlade*> theBlades;
  vector<const GeometricSearchDet*> theComps;
  int              theHelicity;    
};


#endif 
