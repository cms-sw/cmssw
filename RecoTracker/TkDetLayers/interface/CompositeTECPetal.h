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
  virtual const vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const vector<const GeometricSearchDet*>& components() const {return theComps;}
  
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

  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& rod, float window) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;

  int findBin( float R,int layer) const;
  
  GlobalPoint findPosition(int index,int diskSectorIndex) const ;

  const vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontComps : theBackComps);
  }


 private:
  vector<const GeometricSearchDet*> theComps;
  vector<const GeometricSearchDet*> theFrontComps;
  vector<const GeometricSearchDet*> theBackComps;
  vector<const GeomDet*> theBasicComps;

  ReferenceCountingPointer<BoundDiskSector> theFrontSector;
  ReferenceCountingPointer<BoundDiskSector> theBackSector;  
  
};


#endif 
