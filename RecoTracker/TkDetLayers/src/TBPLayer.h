#ifndef TkDetLayers_TBPLayer_h
#define TkDetLayers_TBPLayer_h


#include "TBLayer.h"
#include "PixelRod.h"
#include "TOBRod.h"
#include "Phase2OTBarrelRod.h"

#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

/** A concrete implementation for TOB layer or PixelBarrel layer or a Phase 2 OT Barrel layer as long as it is similar to the previous ones
 *  
 */
#pragma GCC visibility push(hidden)
class TBPLayer GCC11_FINAL : public TBLayer {
 public:
  typedef PeriodicBinFinderInPhi<float>   BinFinderType;


  TBPLayer(std::vector<const PixelRod*>& inner,
	   std::vector<const PixelRod*>& outer) __attribute__ ((cold)):  
    TBLayer(inner,outer, GeomDetEnumerators::PixelBarrel){construct();}
  
  TBPLayer(std::vector<const TOBRod*>& inner,
	   std::vector<const TOBRod*>& outer) __attribute__ ((cold)):  
    TBLayer(inner,outer, GeomDetEnumerators::TOB){construct();}

  TBPLayer(std::vector<const Phase2OTBarrelRod*>& inner,
	   std::vector<const Phase2OTBarrelRod*>& outer) __attribute__ ((cold)):  
    TBLayer(inner,outer, GeomDetEnumerators::P2OTB){construct();}

  
  ~TBPLayer()  __attribute__ ((cold));

  
 

 private:
  // private methods for the implementation of groupedCompatibleDets()

  void construct()  __attribute__ ((cold));


  std::tuple<bool,int,int>  computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const  __attribute__ ((hot));
  


  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const __attribute__ ((hot));
  
  static float calculatePhiWindow( float Xmax, const GeomDet& det,
			     const TrajectoryStateOnSurface& state) __attribute__ ((hot));


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const __attribute__ ((hot));


  BinFinderType    theInnerBinFinder;
  BinFinderType    theOuterBinFinder;

  
  BoundCylinder* cylinder( const std::vector<const GeometricSearchDet*>& rods) const __attribute__ ((cold));

    
};


#pragma GCC visibility pop
#endif 
