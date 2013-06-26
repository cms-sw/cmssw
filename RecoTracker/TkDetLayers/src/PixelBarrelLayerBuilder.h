#ifndef TkDetLayers_PixelBarrelLayerBuilder_h
#define TkDetLayers_PixelBarrelLayerBuilder_h


#include "PixelBarrelLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for PixelBarrelLayer 
 */

#pragma GCC visibility push(hidden)
class PixelBarrelLayerBuilder {  
 public:
  PixelBarrelLayerBuilder(){};
  PixelBarrelLayer* build(const GeometricDet* aPixelBarrelLayer,
			  const TrackerGeometry* theGeomDetGeometry);

  
};


#pragma GCC visibility pop
#endif 
