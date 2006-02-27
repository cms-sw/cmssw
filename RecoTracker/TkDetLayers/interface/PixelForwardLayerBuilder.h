#ifndef TkDetLayers_PixelForwardLayerBuilder_h
#define TkDetLayers_PixelForwardLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for PixelForwardLayer 
 */
using namespace edm;
using namespace std;

class PixelForwardLayerBuilder {  
 public:
  PixelForwardLayerBuilder(){};
  PixelForwardLayer* build(const GeometricDet* aPixelForwardLayer,
			   const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
