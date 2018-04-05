#ifndef TkDetLayers_CompositeTECWedge_h
#define TkDetLayers_CompositeTECWedge_h


#include "TECWedge.h"
#include "SubLayerCrossings.h"


/** A concrete implementation for TEC layer 
 *  built out of TECPetals
 */

#pragma GCC visibility push(hidden)
class CompositeTECWedge final : public TECWedge{
 public:
  CompositeTECWedge(std::vector<const GeomDet*>& innerDets,
		    std::vector<const GeomDet*>& outerDets)  __attribute__ ((cold));

  ~CompositeTECWedge()  override __attribute__ ((cold));
  
  // GeometricSearchDet interface
  const std::vector<const GeomDet*>& basicComponents() const override {return theDets;}

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__ ((cold));
  
  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const override __attribute__ ((cold));

  void
  groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
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
  
  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const __attribute__ ((hot));
  
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
