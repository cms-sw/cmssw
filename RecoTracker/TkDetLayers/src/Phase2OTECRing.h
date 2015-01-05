#ifndef TkDetLayers_Phase2OTECRing_h
#define TkDetLayers_Phase2OTECRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "SubLayerCrossings.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

/** A concrete implementation for TID rings 
 */

#pragma GCC visibility push(hidden)
class Phase2OTECRing GCC11_FINAL : public GeometricSearchDet {
 public:
  Phase2OTECRing(std::vector<const GeomDet*>& innerDets,
		 std::vector<const GeomDet*>& outerDets,
		 std::vector<const GeomDet*>& innerDetBrothers,
		 std::vector<const GeomDet*>& outerDetBrothers);
  ~Phase2OTECRing();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theDisk;}
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold));

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&, 
		       const MeasurementEstimator&) const;

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));
  
 
  //Extension of interface
  virtual const BoundDisk& specificSurface() const {return *theDisk;}
  

 private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& tsos,
				      PropagationDirection propDir) const __attribute__ ((hot));
  
  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   std::vector<DetGroup>& result,
		   std::vector<DetGroup>& brotherresult) const __attribute__ ((hot));

  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			std::vector<DetGroup>& brotherresult,
			bool checkClosest) const __attribute__ ((hot));

  const std::vector<const GeomDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }

  const std::vector<const GeomDet*>& subLayerBrothers( int ind) const {
    return (ind==0 ? theFrontDetBrothers : theBackDetBrothers);
  }


 private:
  std::vector<const GeomDet*> theDets;
  std::vector<const GeomDet*> theFrontDets;
  std::vector<const GeomDet*> theBackDets;
  std::vector<const GeomDet*> theFrontDetBrothers;
  std::vector<const GeomDet*> theBackDetBrothers;

  ReferenceCountingPointer<BoundDisk> theDisk;
  ReferenceCountingPointer<BoundDisk> theFrontDisk;
  ReferenceCountingPointer<BoundDisk> theBackDisk;

  typedef PeriodicBinFinderInPhi<float>   BinFinderType;

  BinFinderType theFrontBinFinder;
  BinFinderType theBackBinFinder;


  
  };


#pragma GCC visibility pop
#endif 
