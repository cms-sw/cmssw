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
class TECLayer : public ForwardDetLayer , public GeometricSearchDetWithGroups {
 public:
  TECLayer(std::vector<const TECPetal*>& innerPetals,
	   std::vector<const TECPetal*>& outerPetals);
  ~TECLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;
 
  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::TEC;}
  

  
 private:
  // private methods for the implementation of groupedCompatibleDets()
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


#pragma GCC visibility pop
#endif 
