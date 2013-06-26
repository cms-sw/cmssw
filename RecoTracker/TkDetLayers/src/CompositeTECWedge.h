#ifndef TkDetLayers_CompositeTECWedge_h
#define TkDetLayers_CompositeTECWedge_h


#include "TECWedge.h"
#include "SubLayerCrossings.h"


/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

#pragma GCC visibility push(hidden)
class CompositeTECWedge GCC11_FINAL : public TECWedge{
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

  virtual void
  groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
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
  
  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const;
  
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


#pragma GCC visibility pop
#endif 
