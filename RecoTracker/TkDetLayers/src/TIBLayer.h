#ifndef TkDetLayers_TIBLayer_h
#define TkDetLayers_TIBLayer_h

#include "TBLayer.h"
#include "TIBRing.h"
#include "TrackingTools/DetLayers/interface/GeneralBinFinderInZforGeometricSearchDet.h"

/** A concrete implementation for TIB layer 
 *  built out of TIBRings
 */

#pragma GCC visibility push(hidden)
class TIBLayer final : public TBLayer {
 public:

  TIBLayer(std::vector<const TIBRing*>& innerRings,
	   std::vector<const TIBRing*>& outerRings) __attribute__ ((cold));

  ~TIBLayer() override __attribute__ ((cold));

 private:
  // private methods for the implementation of groupedCompatibleDets()

  std::tuple<bool,int,int>  computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const  override __attribute__ ((hot));


  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const override __attribute__ ((hot));

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const  override __attribute__ ((hot));

  static bool overlap( const GlobalPoint& gpos, const GeometricSearchDet& ring, float window)   __attribute__ ((hot));


  GeneralBinFinderInZforGeometricSearchDet<float> theInnerBinFinder;
  GeneralBinFinderInZforGeometricSearchDet<float> theOuterBinFinder;

  BoundCylinder* cylinder( const std::vector<const GeometricSearchDet*>& rings) __attribute__ ((cold));


};


#pragma GCC visibility pop
#endif 
