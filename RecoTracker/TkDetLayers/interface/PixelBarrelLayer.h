#ifndef TkDetLayers_PixelBarrelLayer_h
#define TkDetLayers_PixelBarrelLayer_h


#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelRod.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"



/** A concrete implementation for PixelBarrel layer 
 *  built out of BarrelPixelRod
 */

class PixelBarrelLayer : public RodBarrelLayer{
 public:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;


  PixelBarrelLayer(vector<const PixelRod*>& innerRods,
		   vector<const PixelRod*>& outerRods);
  
  ~PixelBarrelLayer();
  
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
  virtual Module   module()   const { return pixel;}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  // the implementation of the methods is the same of the TOBLayer one.
  // In the future, to move common code in a common place!

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

  bool overlap( const GlobalPoint& gpos, const PixelRod& rod, float phiWin) const;


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			vector<DetGroup>& result,
			bool checkClosest) const;

  const vector<const PixelRod*>& subLayer( int ind) const {
    return (ind==0 ? theInnerRods : theOuterRods);}
  
  BoundCylinder* cylinder( const vector<const PixelRod*>& rods) const ;


 private:
  vector<const PixelRod*> theRods;
  vector<const GeometricSearchDet*> theComponents;
  vector<const PixelRod*> theInnerRods;
  vector<const PixelRod*> theOuterRods;

  BinFinderType    theInnerBinFinder;
  BinFinderType    theOuterBinFinder;

  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;

  
};


#endif 
