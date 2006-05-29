#ifndef TkDetLayers_TECLayer_h
#define TkDetLayers_TECLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/TECPetal.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
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
  
  virtual const vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const vector<const GeometricSearchDet*>& components() const {return theComps;}
  
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
  

  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& petal, float window) const;

  const vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontComps : theBackComps);
  }


 protected:
  virtual BoundDisk* computeDisk( vector<const GeometricSearchDet*>& petals) const;

  vector<const GeometricSearchDet*> theComps;
  vector<const GeometricSearchDet*> theFrontComps;
  vector<const GeometricSearchDet*> theBackComps;
  vector<const GeomDet*> theBasicComps;


  ReferenceCountingPointer<BoundDisk>  theFrontDisk;
  ReferenceCountingPointer<BoundDisk>  theBackDisk;

  typedef PeriodicBinFinderInPhi<double>   BinFinderPhi;

  BinFinderPhi theFrontBinFinder;
  BinFinderPhi theBackBinFinder;

  
};


#endif 
