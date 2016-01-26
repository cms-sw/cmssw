#ifndef TkDetLayers_Phase2OTtiltedBarrelLayer_h
#define TkDetLayers_Phase2OTtiltedBarrelLayer_h


//#include "Phase2OTBarrelLayer.h"
#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "Phase2OTBarrelLayer.h"
#include "Phase2OTBarrelRod.h"
#include "Phase2OTEndcapRing.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "SubLayerCrossings.h"



/** A concrete implementation for Phase2OTBarrel layer 
 *  built out of BarrelPhase2OTBarrelRod
 */

#pragma GCC visibility push(hidden)
//class Phase2OTtiltedBarrelLayer : public RodBarrelLayer, public GeometricSearchDetWithGroups {
class Phase2OTtiltedBarrelLayer GCC11_FINAL : public Phase2OTBarrelLayer {
 public:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;

  Phase2OTtiltedBarrelLayer(std::vector<const Phase2OTBarrelRod*>& innerRods,
		            std::vector<const Phase2OTBarrelRod*>& outerRods,
                            std::vector<const Phase2OTEndcapRing*>& rings);
  
  ~Phase2OTtiltedBarrelLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
/*
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;
    
*/
  // DetLayer interface
  //virtual SubDetector subDetector() const { return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2OTB];}

/*
 private:
  // private methods for the implementation of groupedCompatibleDets()
  // the implementation of the methods is the same of the TOBLayer one.
  // In the future, to move common code in a common place!

  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& tsos,
				      PropagationDirection propDir) const;
  
  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   std::vector<DetGroup>& result) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
  double calculatePhiWindow( double Xmax, const GeomDet& det,
			     const TrajectoryStateOnSurface& state) const;

  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& rod, float phiWin) const;


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const;

  const std::vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theInnerComps : theOuterComps);}
  
  BoundCylinder* cylinder( const std::vector<const GeometricSearchDet*>& rods) const ;

*/
 private:
  //Phase2OTBarrelLayer thePhase2OTBarrelLayer;
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeometricSearchDet*> theInnerRodsComps;
  std::vector<const GeometricSearchDet*> theOuterRodsComps;
  std::vector<const GeometricSearchDet*> theRingsComps;
  std::vector<const GeomDet*> theBasicComps;

  BinFinderType    theInnerBinFinder;
  BinFinderType    theOuterBinFinder;

  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;

  
};


#pragma GCC visibility pop
#endif 
