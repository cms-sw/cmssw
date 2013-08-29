#ifndef TkDetLayers_PixelBlade_h
#define TkDetLayers_PixelBlade_h

#include "BoundDiskSector.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"
#include "SubLayerCrossings.h"


/** A concrete implementation for PixelBlade
 */

#pragma GCC visibility push(hidden)
class PixelBlade GCC11_FINAL : public GeometricSearchDetWithGroups {
 public:

  PixelBlade(std::vector<const GeomDet*>& frontDets,
	     std::vector<const GeomDet*>& backDets  );

  ~PixelBlade();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theDiskSector;}

  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}

  virtual const std::vector<const GeometricSearchDet*>& components() const;

  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;
  
  virtual void 
  groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			  const Propagator& prop,
			  const MeasurementEstimator& est,
			  std::vector<DetGroup> & result) const;
  
  //Extension of the interface
  virtual const BoundDiskSector& specificSurface() const {return *theDiskSector;}

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

  bool overlap( const GlobalPoint& gpos, const GeomDet& det, float phiWin) const;

  // This 2 find methods should be substituted with the use 
  // of a GeneralBinFinderInR
  
  int findBin( float R,int layer) const;
  int findBin2( GlobalPoint thispoint,int layer) const;
  
  GlobalPoint findPosition(int index,int diskSectorIndex) const ;

  const std::vector<const GeomDet*>& subBlade( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }



 private:
  std::vector<const GeomDet*> theDets;
  std::vector<const GeomDet*> theFrontDets;
  std::vector<const GeomDet*> theBackDets;
  
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;
  ReferenceCountingPointer<BoundDiskSector> theFrontDiskSector;
  ReferenceCountingPointer<BoundDiskSector> theBackDiskSector;
};


#pragma GCC visibility pop
#endif 
