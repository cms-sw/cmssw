#ifndef TkDetLayers_TOBRod_h
#define TkDetLayers_TOBRod_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"
#include "SubLayerCrossings.h"


/** A concrete implementation for TOB Rod 
 *  
 */

#pragma GCC visibility push(hidden)
class TOBRod final : public DetRod {
 public:
  typedef PeriodicBinFinderInZ<float>   BinFinderType;

  TOBRod(std::vector<const GeomDet*>& innerDets,
	 std::vector<const GeomDet*>& outerDets) __attribute__ ((cold));
  ~TOBRod() override __attribute__ ((cold));
  
  // GeometricSearchDet interface
  
  const std::vector<const GeomDet*>& basicComponents() const override {return theDets;}

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__ ((cold));

  
  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const  override __attribute__ ((cold));

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const override __attribute__ ((hot));
  
 
 private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& tsos,
				      PropagationDirection propDir) const __attribute__ ((hot));
  
  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   std::vector<DetGroup>& result) const __attribute__ ((hot));

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const __attribute__ ((hot));


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const __attribute__ ((hot));


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
