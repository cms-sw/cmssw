#ifndef TkDetLayers_TIDRing_h
#define TkDetLayers_TIDRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInPhi.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"
#include "Geometry/Surface/interface/BoundDisk.h"

/** A concrete implementation for TID rings 
 */

class TIDRing : public GeometricSearchDet{
 public:
  TIDRing(vector<const GeomDet*>& innerDets,
	  vector<const GeomDet*>& outerDets);
  ~TIDRing();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theDisk;}
  
  virtual vector<const GeomDet*> basicComponents() const;
  
  virtual vector<const GeometricSearchDet*> components() const {
    return vector<const GeometricSearchDet*>();}
    

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

  //Extension of interface
  virtual const BoundDisk& specificSurface() const {return *theDisk;}
  

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

  float calculatePhiWindow( const MeasurementEstimator::Local2DVector&  maxDistance, 
  			    const TrajectoryStateOnSurface& ts, 
			    const BoundPlane& plane) const;

  pair<float, float> computeDetPhiRange( const BoundPlane& plane) const;
  

  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			vector<DetGroup>& result,
			bool checkClosest) const;

  bool overlapInPhi( const GlobalPoint& startPoint,const GeomDet* det, float phiWin ) const;  

  const vector<const GeomDet*>& subLayer( int ind) const {
    return (ind==0 ? theFrontDets : theBackDets);
  }


 private:
  vector<const GeomDet*> theDets;
  vector<const GeomDet*> theFrontDets;
  vector<const GeomDet*> theBackDets;

  ReferenceCountingPointer<BoundDisk> theDisk;
  ReferenceCountingPointer<BoundDisk> theFrontDisk;
  ReferenceCountingPointer<BoundDisk> theBackDisk;

  typedef PeriodicBinFinderInPhi<double>   BinFinderType;

  BinFinderType theFrontBinFinder;
  BinFinderType theBackBinFinder;


  
  };


#endif 
