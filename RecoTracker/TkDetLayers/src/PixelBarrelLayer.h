#ifndef TkDetLayers_PixelBarrelLayer_h
#define TkDetLayers_PixelBarrelLayer_h


#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "PixelRod.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "SubLayerCrossings.h"



/** A concrete implementation for PixelBarrel layer 
 *  built out of BarrelPixelRod
 */

#pragma GCC visibility push(hidden)
class PixelBarrelLayer GCC11_FINAL : public RodBarrelLayer, public GeometricSearchDetWithGroups {
 public:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;


  PixelBarrelLayer(std::vector<const PixelRod*>& innerRods,
		   std::vector<const PixelRod*>& outerRods);
  
  ~PixelBarrelLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;
    

  // DetLayer interface
  virtual SubDetector subDetector() const { return GeomDetEnumerators::PixelBarrel;}


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


 private:
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeometricSearchDet*> theInnerComps;
  std::vector<const GeometricSearchDet*> theOuterComps;
  std::vector<const GeomDet*> theBasicComps;

  BinFinderType    theInnerBinFinder;
  BinFinderType    theOuterBinFinder;

  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;

  
};


#pragma GCC visibility pop
#endif 
