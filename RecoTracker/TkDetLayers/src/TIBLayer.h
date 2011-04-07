#ifndef TkDetLayers_TIBLayer_h
#define TkDetLayers_TIBLayer_h


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TIBRing.h"
#include "SubLayerCrossings.h"
#include "TrackingTools/DetLayers/interface/GeneralBinFinderInZforGeometricSearchDet.h"

/** A concrete implementation for TIB layer 
 *  built out of TIBRings
 */

#pragma GCC visibility push(hidden)
class TIBLayer : public BarrelDetLayer, public GeometricSearchDetWithGroups {
 public:

  TIBLayer(std::vector<const TIBRing*>& innerRings,
	   std::vector<const TIBRing*>& outerRings);

  ~TIBLayer();
  
  // GeometricSearchDet interface

  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;
 
  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::TIB;}


 private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& startingState,
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

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;

  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& ring, float window) const;

  const std::vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theInnerComps : theOuterComps);
  }


 private:
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeometricSearchDet*> theInnerComps;
  std::vector<const GeometricSearchDet*> theOuterComps;
  std::vector<const GeomDet*> theBasicComps;
  
  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;

  GeneralBinFinderInZforGeometricSearchDet<float> theInnerBinFinder;
  GeneralBinFinderInZforGeometricSearchDet<float> theOuterBinFinder;

  BoundCylinder* cylinder( const std::vector<const GeometricSearchDet*>& rings);


};


#pragma GCC visibility pop
#endif 
