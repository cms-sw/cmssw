#ifndef TkDetLayers_TIBLayer_h
#define TkDetLayers_TIBLayer_h


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBRing.h"
#include "RecoTracker/TkDetLayers/interface/SubLayerCrossings.h"
#include "TrackingTools/DetLayers/interface/GeneralBinFinderInZforGeometricSearchDet.h"

/** A concrete implementation for TIB layer 
 *  built out of TIBRings
 */

class TIBLayer : public BarrelDetLayer {
 public:

  TIBLayer(vector<const TIBRing*>& innerRings,
	   vector<const TIBRing*>& outerRings);

  ~TIBLayer();
  
  // GeometricSearchDet interface

  virtual vector<const GeomDet*> basicComponents() const;

  virtual vector<const GeometricSearchDet*> components() const;
  
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<GeometricSearchDet::DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& tsos,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const; 
    //{return TkGeometricSearchDet::compatibleDets(tsos,prop,est);}

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return true;};  

  // DetLayer interface
  virtual Module   module()   const { return silicon;}

 private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& startingState,
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

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;

  bool overlap( const GlobalPoint& gpos, const TIBRing& ring, float window) const;

  const vector<const TIBRing*>& subLayer( int ind) const {
    return (ind==0 ? theInnerRings : theOuterRings);
  }


 private:
  vector<const TIBRing*> theRings;
  vector<const GeometricSearchDet*> theComponents;
  vector<const TIBRing*> theInnerRings;
  vector<const TIBRing*> theOuterRings;
  
  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;

  GeneralBinFinderInZforGeometricSearchDet<float> theInnerBinFinder;
  GeneralBinFinderInZforGeometricSearchDet<float> theOuterBinFinder;

  BoundCylinder* cylinder( const vector<const TIBRing*>& rings);


};


#endif 
