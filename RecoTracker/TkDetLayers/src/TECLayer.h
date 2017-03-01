#ifndef TkDetLayers_TECLayer_h
#define TkDetLayers_TECLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TECPetal.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "SubLayerCrossings.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

#pragma GCC visibility push(hidden)
class TECLayer : public ForwardDetLayer  {
 public:
  TECLayer(std::vector<const TECPetal*>& innerPetals,
	   std::vector<const TECPetal*>& outerPetals) __attribute__ ((cold));
  ~TECLayer() __attribute__ ((cold));
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold)) {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));
 
  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::TEC];}
  

  
 private:


  // private methods for the implementation of groupedCompatibleDets()
  SubLayerCrossings   computeCrossings( const TrajectoryStateOnSurface& startingState,
					PropagationDirection propDir) const __attribute__ ((hot));

  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   std::vector<DetGroup>& result) const __attribute__ ((hot));

  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const __attribute__ ((hot));
  


  const std::vector<const TECPetal*>& subLayer( int ind) const {
    return (ind==0 ? theFrontComps : theBackComps);
  }


 protected:

  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeomDet*> theBasicComps;

  std::vector<const TECPetal*> theFrontComps;
  std::vector<const TECPetal*> theBackComps;
 

  ReferenceCountingPointer<BoundDisk>  theFrontDisk;
  ReferenceCountingPointer<BoundDisk>  theBackDisk;

  typedef PeriodicBinFinderInPhi<float>   BinFinderPhi;

  BinFinderPhi theFrontBinFinder;
  BinFinderPhi theBackBinFinder;

  
};


#pragma GCC visibility pop
#endif 
