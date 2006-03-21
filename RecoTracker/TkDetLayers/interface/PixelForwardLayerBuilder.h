#ifndef TkDetLayers_PixelForwardLayerBuilder_h
#define TkDetLayers_PixelForwardLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for PixelForwardLayer 
 */
using namespace edm;
using namespace std;

class PixelForwardLayerBuilder {  
 public:
  PixelForwardLayerBuilder(){};
  PixelForwardLayer* build(const GeometricDet* aPixelForwardLayer,
			   const TrackerGeometry* theGeomDetGeometry);

  
};


#endif 
