#ifndef TkDetLayers_TECLayer_h
#define TkDetLayers_TECLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/TECPetal.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInPhi.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

class TECLayer : public ForwardDetLayer{
 public:
  TECLayer(vector<const TECPetal*>& innerPetals,
	   vector<const TECPetal*>& outerPetals);
  ~TECLayer();
  
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

  // DetLayer interface
  virtual Module   module()   const { return silicon;}

  

  
 private:
  // private methods for the implementation of groupedCompatibleDets()
  float calculatePhiWindow(const MeasurementEstimator::Local2DVector& maxDistance,
			   const TrajectoryStateOnSurface& ts, 
			   const BoundPlane& plane) const;
  
  SubLayerCrossings   computeCrossings( const TrajectoryStateOnSurface& startingState,
					PropagationDirection propDir) const;

  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   vector<DetGroup>& result) const;

  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			vector<DetGroup>& result,
			bool checkClosest) const;

  
  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  

  bool overlap( const GlobalPoint& gpos, const TECPetal& petal, float window) const;

  const vector<const TECPetal*>& subLayer( int ind) const {
    return (ind==0 ? theFrontPetals : theBackPetals);
  }


 protected:
  virtual BoundDisk* computeDisk( vector<const TECPetal*>& petals) const;

  vector<const TECPetal*> thePetals;
  vector<const TECPetal*> theFrontPetals;
  vector<const TECPetal*> theBackPetals;

  ReferenceCountingPointer<BoundDisk>  theLayerDisk;
  ReferenceCountingPointer<BoundDisk>  theFrontDisk;
  ReferenceCountingPointer<BoundDisk>  theBackDisk;

  typedef PeriodicBinFinderInPhi<double>   BinFinderPhi;

  BinFinderPhi theFrontBinFinder;
  BinFinderPhi theBackBinFinder;

  
};


#endif 
