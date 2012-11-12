#ifndef TkDetLayers_TIBRing_h
#define TkDetLayers_TIBRing_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

/** A concrete implementation for TIB rings 
 */

#pragma GCC visibility push(hidden)
class TIBRing : public GeometricSearchDetWithGroups{
 public:
  TIBRing(std::vector<const GeomDet*>& theGeomDets);
  ~TIBRing();
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theCylinder;}  

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

 
  //--- Extension of the interface
  
  /// Return the ring surface as a 
  virtual const BoundCylinder& specificSurface() const {return *theCylinder;}
 
 private:
  //general private methods

  void checkPeriodicity(std::vector<const GeomDet*>::const_iterator first,
			std::vector<const GeomDet*>::const_iterator last);

  void checkRadius(std::vector<const GeomDet*>::const_iterator first,
		   std::vector<const GeomDet*>::const_iterator last);
  
  void computeHelicity();

  // methods for groupedCompatibleDets implementation
  struct SubRingCrossings {
    SubRingCrossings():isValid_(false){};
    SubRingCrossings( int ci, int ni, float nd) : 
      isValid_(true),closestIndex(ci), nextIndex(ni), nextDistance(nd) {}
    
    bool  isValid_;
    int   closestIndex;
    int   nextIndex;
    float nextDistance;
  };


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubRingCrossings& crossings,
			float window, 
			std::vector<DetGroup>& result) const;

  SubRingCrossings 
  computeCrossings( const TrajectoryStateOnSurface& startingState,
		    PropagationDirection propDir) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;




 private:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;
  BinFinderType    theBinFinder;

  std::vector<const GeomDet*> theDets;
  ReferenceCountingPointer<BoundCylinder>  theCylinder;
  int              theHelicity;    
};


#pragma GCC visibility pop
#endif 
