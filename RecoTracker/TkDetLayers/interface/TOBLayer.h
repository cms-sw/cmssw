#ifndef TkDetLayers_TOBLayer_h
#define TkDetLayers_TOBLayer_h


#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "RecoTracker/TkDetLayers/interface/TOBRod.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"

/** A concrete implementation for TOB layer 
 *  built out of TOBRods
 */

class TOBLayer : public RodBarrelLayer{
 public:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;


  TOBLayer(vector<const TOBRod*>& innerRods,
	   vector<const TOBRod*>& outerRods);
  ~TOBLayer();
  
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

  // DetLayer interface
  virtual Module   module()   const { return silicon;}
  
 

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
  
  double calculatePhiWindow( double Xmax, const GeomDet& det,
			     const TrajectoryStateOnSurface& state) const;

  bool overlap( const GlobalPoint& gpos, const TOBRod& rod, float phiWin) const;


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			vector<DetGroup>& result,
			bool checkClosest) const;

  const vector<const TOBRod*>& subLayer( int ind) const {
    return (ind==0 ? theInnerRods : theOuterRods);}
  
  BoundCylinder* cylinder( const vector<const TOBRod*>& rods) const ;


 private:
  vector<const TOBRod*> theRods;
  vector<const GeometricSearchDet*> theComponents;
  vector<const TOBRod*> theInnerRods;
  vector<const TOBRod*> theOuterRods;
  
  BinFinderType    theInnerBinFinder;
  BinFinderType    theOuterBinFinder;

  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;
    
};


#endif 
