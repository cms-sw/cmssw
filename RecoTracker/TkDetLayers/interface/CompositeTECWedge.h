#ifndef TkDetLayers_CompositeTECWedge_h
#define TkDetLayers_CompositeTECWedge_h


#include "RecoTracker/TkDetLayers/interface/TECWedge.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"


/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

class CompositeTECWedge : public TECWedge{
 public:
  CompositeTECWedge(vector<const GeomDet*>& innerDets,
		    vector<const GeomDet*>& outerDets);

  ~CompositeTECWedge();
  
  // GeometricSearchDet interface
  virtual vector<const GeomDet*> basicComponents() const;

  virtual vector<const GeometricSearchDet*> components() const;
  
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<DetGroup> 
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
		   vector<DetGroup>& result) const;
  
  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			vector<DetGroup>& result,
			bool checkClosest) const;

  bool overlap( const GlobalPoint& gpos, const GeomDet& det, float window) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;

  float calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
			    const TrajectoryStateOnSurface& ts, 
			    const BoundPlane& plane) const;

  pair<float, float> computeDetPhiRange( const BoundPlane& plane) const;

  int findClosestDet( const GlobalPoint& startPos,int sectorId) const;

  const vector<const GeomDet*>& subWedge( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }


 private:
  vector<const GeomDet*> theFrontDets;
  vector<const GeomDet*> theBackDets;
  vector<const GeomDet*> theDets;
  
  ReferenceCountingPointer<BoundDiskSector>  theFrontSector;
  ReferenceCountingPointer<BoundDiskSector>  theBackSector;
  
};


#endif 
