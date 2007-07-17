#ifndef TkDetLayers_CompositeTECPetal_h
#define TkDetLayers_CompositeTECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "RecoTracker/TkDetLayers/interface/TECPetal.h"
#include "RecoTracker/TkDetLayers/interface/TECWedge.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"


/** A concrete implementation for TEC petals
 */

class CompositeTECPetal : public TECPetal{
 public:
  CompositeTECPetal(std::vector<const TECWedge*>& innerWedges,
		    std::vector<const TECWedge*>& outerWedges);

  ~CompositeTECPetal();
  
  // GeometricSearchDet interface  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
  
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
  SubLayerCrossings computeCrossings(const TrajectoryStateOnSurface& tsos,
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

  static
  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& rod, float window);

  static
  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est);

  int findBin( float R,int layer) const;
  
  GlobalPoint findPosition(int index,int diskSectorIndex) const ;

  const std::vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontComps : theBackComps);
  }


 private:
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeometricSearchDet*> theFrontComps;
  std::vector<const GeometricSearchDet*> theBackComps;
  std::vector<const GeomDet*> theBasicComps;

  std::vector<float> theFrontBoundaries;
  std::vector<float> theBackBoundaries;

  ReferenceCountingPointer<BoundDiskSector> theFrontSector;
  ReferenceCountingPointer<BoundDiskSector> theBackSector;  
  
};


#endif 
