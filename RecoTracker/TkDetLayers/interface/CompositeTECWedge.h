#ifndef TkDetLayers_CompositeTECWedge_h
#define TkDetLayers_CompositeTECWedge_h


#include "RecoTracker/TkDetLayers/interface/TECWedge.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"


/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

class CompositeTECWedge : public TECWedge{
 public:
  CompositeTECWedge(std::vector<const GeomDet*>& innerDets,
		    std::vector<const GeomDet*>& outerDets);

  ~CompositeTECWedge();
  
  // GeometricSearchDet interface
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theDets;}

  virtual const std::vector<const GeometricSearchDet*>& components() const;
  
  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;

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

  bool overlap( const GlobalPoint& gpos, const GeomDet& det, float window) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;

  float calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
			    const TrajectoryStateOnSurface& ts, 
			    const BoundPlane& plane) const;

  std::pair<float, float> computeDetPhiRange( const BoundPlane& plane) const;

  int findClosestDet( const GlobalPoint& startPos,int sectorId) const;

  const std::vector<const GeomDet*>& subWedge( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }


 private:
  std::vector<const GeomDet*> theFrontDets;
  std::vector<const GeomDet*> theBackDets;
  std::vector<const GeomDet*> theDets;
  
  ReferenceCountingPointer<BoundDiskSector>  theFrontSector;
  ReferenceCountingPointer<BoundDiskSector>  theBackSector;
  
};


#endif 
