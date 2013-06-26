#ifndef TkDetLayers_PixelForwardLayerBuilder_h
#define TkDetLayers_PixelForwardLayerBuilder_h


#include "PixelForwardLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for PixelForwardLayer 
 */

#pragma GCC visibility push(hidden)
class PixelForwardLayerBuilder {  
 public:
  PixelForwardLayerBuilder(){};
  ForwardDetLayer* build(const GeometricDet* aPixelForwardLayer,
			   const TrackerGeometry* theGeomDetGeometry);

  
};


#pragma GCC visibility pop
#endif 
