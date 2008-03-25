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
  TECLayer(std::vector<const TECPetal*>& innerPetals,
	   std::vector<const TECPetal*>& outerPetals);
  ~TECLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
  
  virtual std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return true;}  

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::TEC;}
  

  
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
		   std::vector<DetGroup>& result) const;

  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const;

  
  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  

  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& petal, float window) const;

  const std::vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontComps : theBackComps);
  }


 protected:
  virtual BoundDisk* computeDisk( std::vector<const GeometricSearchDet*>& petals) const;

  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeometricSearchDet*> theFrontComps;
  std::vector<const GeometricSearchDet*> theBackComps;
  std::vector<const GeomDet*> theBasicComps;


  ReferenceCountingPointer<BoundDisk>  theFrontDisk;
  ReferenceCountingPointer<BoundDisk>  theBackDisk;

  typedef PeriodicBinFinderInPhi<double>   BinFinderPhi;

  BinFinderPhi theFrontBinFinder;
  BinFinderPhi theBackBinFinder;

  
};


#endif 
