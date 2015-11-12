#ifndef TkDetLayers_Phase2OTBarrelRod_h
#define TkDetLayers_Phase2OTBarrelRod_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "Utilities/BinningTools/interface/GenericBinFinderInZ.h"
#include "SubLayerCrossings.h"


/** A concrete implementation for TOB Rod 
 *  
 */

#pragma GCC visibility push(hidden)
class Phase2OTBarrelRod GCC11_FINAL : public DetRod, public GeometricSearchDetWithGroups{
 public:
  typedef GenericBinFinderInZ<float,GeomDet>   BinFinderType;

  Phase2OTBarrelRod(std::vector<const GeomDet*>& innerDets,
		    std::vector<const GeomDet*>& outerDets);
  ~Phase2OTBarrelRod();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}

  virtual const std::vector<const GeometricSearchDet*>& components() const;

  
  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;
  
 
 private:
  // private methods for the implementation of groupedCompatibleDets()

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


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const;

  bool overlap( const GlobalPoint& gpos, const GeomDet& rod, float phiWin) const;

  const std::vector<const GeomDet*>& subRod( int ind) const {
    return (ind==0 ? theInnerDets : theOuterDets);
  }


 private:
  std::vector<const GeomDet*> theDets;
  std::vector<const GeomDet*> theInnerDets;
  std::vector<const GeomDet*> theOuterDets;

  ReferenceCountingPointer<Plane> theInnerPlane;
  ReferenceCountingPointer<Plane> theOuterPlane;

  BinFinderType theInnerBinFinder;
  BinFinderType theOuterBinFinder;

};


#pragma GCC visibility pop
#endif 
