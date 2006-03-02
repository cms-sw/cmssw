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
  CompositeTECPetal(vector<const TECWedge*>& innerWedges,
		    vector<const TECWedge*>& outerWedges);

  ~CompositeTECPetal();
  
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
  SubLayerCrossings computeCrossings(const TrajectoryStateOnSurface& tsos,
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

  bool overlap( const GlobalPoint& gpos, const TECWedge& rod, float window) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;

  int findBin( float R,int layer) const;
  
  GlobalPoint findPosition(int index,int diskSectorType) const ;

  const vector<const TECWedge*>& subLayer( int ind) const {
    return (ind==0 ? theFrontWedges : theBackWedges);
  }


 private:
  vector<const TECWedge*> theWedges;
  vector<const TECWedge*> theFrontWedges;
  vector<const TECWedge*> theBackWedges;

  ReferenceCountingPointer<BoundDiskSector> theFrontSector;
  ReferenceCountingPointer<BoundDiskSector> theBackSector;  
  
};


#endif 
