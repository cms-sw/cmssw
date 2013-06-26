#ifndef TkDetLayers_TIDRing_h
#define TkDetLayers_TIDRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "SubLayerCrossings.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

/** A concrete implementation for TID rings 
 */

#pragma GCC visibility push(hidden)
class TIDRing GCC11_FINAL : public GeometricSearchDetWithGroups{
 public:
  TIDRing(std::vector<const GeomDet*>& innerDets,
	  std::vector<const GeomDet*>& outerDets);
  ~TIDRing();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theDisk;}
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const;

  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&, 
		       const MeasurementEstimator&) const;

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;
  
 
  //Extension of interface
  virtual const BoundDisk& specificSurface() const {return *theDisk;}
  

 private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& tsos,
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

  const std::vector<const GeomDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }


 private:
  std::vector<const GeomDet*> theDets;
  std::vector<const GeomDet*> theFrontDets;
  std::vector<const GeomDet*> theBackDets;

  ReferenceCountingPointer<BoundDisk> theDisk;
  ReferenceCountingPointer<BoundDisk> theFrontDisk;
  ReferenceCountingPointer<BoundDisk> theBackDisk;

  typedef PeriodicBinFinderInPhi<double>   BinFinderType;

  BinFinderType theFrontBinFinder;
  BinFinderType theBackBinFinder;


  
  };


#pragma GCC visibility pop
#endif 
