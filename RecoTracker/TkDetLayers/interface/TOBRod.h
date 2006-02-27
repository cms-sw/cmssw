#ifndef TkDetLayers_TOBRod_h
#define TkDetLayers_TOBRod_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"


/** A concrete implementation for TOB Rod 
 *  
 */

class TOBRod : public DetRod{
 public:
  typedef PeriodicBinFinderInZ<float>   BinFinderType;

  TOBRod(vector<const GeomDet*>& innerDets,
	 vector<const GeomDet*>& outerDets);
  ~TOBRod();
  
  // GeometricSearchDet interface
  
  virtual vector<const GeomDet*> basicComponents() const;

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

  bool overlap( const GlobalPoint& gpos, const GeomDet& rod, float phiWin) const;

  const vector<const GeomDet*>& subRod( int ind) const {
    return (ind==0 ? theInnerDets : theOuterDets);
  }


 private:
  vector<const GeomDet*> theDets;
  vector<const GeomDet*> theInnerDets;
  vector<const GeomDet*> theOuterDets;

  ReferenceCountingPointer<BoundPlane> theInnerPlane;
  ReferenceCountingPointer<BoundPlane> theOuterPlane;

  BinFinderType theInnerBinFinder;
  BinFinderType theOuterBinFinder;

};


#endif 
