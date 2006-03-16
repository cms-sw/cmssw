#ifndef TkDetLayers_PixelBlade_h
#define TkDetLayers_PixelBlade_h

#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"


/** A concrete implementation for PixelBlade
 */

class PixelBlade : public GeometricSearchDet{
 public:

  PixelBlade(vector<const GeomDet*>& frontDets,
	     vector<const GeomDet*>& backDets  );

  ~PixelBlade(){};
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theDiskSector;}

  virtual vector<const GeomDet*> basicComponents() const {return theDets;}

  virtual vector<const GeometricSearchDet*> components() const;

    
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;

  

  virtual bool hasGroups() const {return true;}

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
		   vector<DetGroup>& result) const;
  
  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			vector<DetGroup>& result,
			bool checkClosest) const;

  bool overlap( const GlobalPoint& gpos, const GeomDet& det, float phiWin) const;

  // This 2 find methods should be substituted with the use 
  // of a GeneralBinFinderInR
  
  int findBin( float R,int layer) const;
  
  GlobalPoint findPosition(int index,int diskSectorIndex) const ;

  const vector<const GeomDet*>& subBlade( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }



 private:
  vector<const GeomDet*> theDets;
  vector<const GeomDet*> theFrontDets;
  vector<const GeomDet*> theBackDets;
  
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;
  ReferenceCountingPointer<BoundDiskSector> theFrontDiskSector;
  ReferenceCountingPointer<BoundDiskSector> theBackDiskSector;
};


#endif 
