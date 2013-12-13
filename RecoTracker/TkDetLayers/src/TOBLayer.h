#ifndef TkDetLayers_TOBLayer_h
#define TkDetLayers_TOBLayer_h


#include "TBLayer.h"
#include "TOBRod.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
/** A concrete implementation for TOB layer 
 *  built out of TOBRods
 */

#pragma GCC visibility push(hidden)
class TOBLayer GCC11_FINAL : public TBLayer {
 public:
  typedef PeriodicBinFinderInPhi<float>   BinFinderType;


  TOBLayer(std::vector<const TOBRod*>& innerRods,
	   std::vector<const TOBRod*>& outerRods)  __attribute__ ((cold));
  ~TOBLayer()  __attribute__ ((cold));
  


  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::TOB;}
 

 private:
  // private methods for the implementation of groupedCompatibleDets()

  std::tuple<bool,int,int>  computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const  __attribute__ ((hot));
  


  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const __attribute__ ((hot));
  
  double calculatePhiWindow( double Xmax, const GeomDet& det,
			     const TrajectoryStateOnSurface& state) const __attribute__ ((hot));

  bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& rod, float phiWin) const  __attribute__ ((hot));


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
