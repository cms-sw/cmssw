#ifndef TkDetLayers_TBLayer_h
#define TkDetLayers_TBLayer_h


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "SubLayerCrossings.h"
#include <tuple>



// A base class for Barrel Layers
#pragma GCC visibility push(hidden)
class TBLayer: public BarrelDetLayer {
 public:
  
  template<typename TDET>
  TBLayer(std::vector<const TDET*>& inner,
	  std::vector<const TDET*>& outer, GeomDetEnumerators::SubDetector ime) :
    BarrelDetLayer(true),
    theInnerComps(inner.begin(),inner.end()), 
    theOuterComps(outer.begin(),outer.end()), me(ime){}


  ~TBLayer() __attribute__ ((cold));

  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const final {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const final  __attribute__ ((cold)) {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const final __attribute__ ((hot));


  // DetLayer interface
  virtual SubDetector subDetector() const final {return GeomDetEnumerators::subDetGeom[me];}


protected:


  SubLayerCrossings computeCrossings( const TrajectoryStateOnSurface& startingState,
				      PropagationDirection propDir) const  __attribute__ ((hot));


  virtual std::tuple<bool,int,int>  computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const=0;

  virtual float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const =0;


  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   std::vector<DetGroup>& result) const __attribute__ ((hot));


 
  const std::vector<const GeometricSearchDet*>& subLayer( int ind) const {
    return (ind==0 ? theInnerComps : theOuterComps);
  }

  bool isTIB() const { return me==GeomDetEnumerators::TIB;}
  bool isTOB() const { return me==GeomDetEnumerators::TOB;}
  bool isPixel() const { return me==GeomDetEnumerators::PixelBarrel;}
  bool isPhase2OT() const { return me==GeomDetEnumerators::P2OTB;}

  virtual void searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
				std::vector<DetGroup>& result,
				bool checkClosest) const=0;


protected:
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeometricSearchDet*> theInnerComps;
  std::vector<const GeometricSearchDet*> theOuterComps;
  std::vector<const GeomDet*> theBasicComps;
  
  ReferenceCountingPointer<BoundCylinder>  theInnerCylinder;
  ReferenceCountingPointer<BoundCylinder>  theOuterCylinder;

  GeomDetEnumerators::SubDetector me;

};


#pragma GCC visibility pop

#endif
