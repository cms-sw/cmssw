#ifndef TkDetLayers_TIBRing_h
#define TkDetLayers_TIBRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInPhi.h"

/** A concrete implementation for TIB rings 
 */

class TIBRing : public GeometricSearchDet{
 public:
  TIBRing(vector<const GeomDet*>& theGeomDets);
  ~TIBRing();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theCylinder;}  

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
 
  //--- Extension of the interface
  
  /// Return the ring surface as a 
  virtual const BoundCylinder& specificSurface() const {return *theCylinder;}
 
 private:
  //general private methods

  void checkPeriodicity(vector<const GeomDet*>::const_iterator first,
			vector<const GeomDet*>::const_iterator last);

  void checkRadius(vector<const GeomDet*>::const_iterator first,
		   vector<const GeomDet*>::const_iterator last);
  
  void computeHelicity();

  // methods for groupedCompatibleDets implementation
  struct SubRingCrossings {
    SubRingCrossings(){};
    SubRingCrossings( int ci, int ni, float nd) : 
      closestIndex(ci), nextIndex(ni), nextDistance(nd) {}
    
    int   closestIndex;
    int   nextIndex;
    float nextDistance;
  };


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubRingCrossings& crossings,
			float window, 
			vector<DetGroup>& result) const;

  SubRingCrossings 
  computeCrossings( const TrajectoryStateOnSurface& startingState,
		    PropagationDirection propDir) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;




 private:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;
  BinFinderType    theBinFinder;

  vector<const GeomDet*> theDets;
  ReferenceCountingPointer<BoundCylinder>  theCylinder;
  int              theHelicity;    
};


#endif 
