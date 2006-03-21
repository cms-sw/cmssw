#ifndef TkDetLayers_PixelBarrelLayerBuilder_h
#define TkDetLayers_PixelBarrelLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for PixelBarrelLayer 
 */
using namespace edm;
using namespace std;

class PixelBarrelLayerBuilder {  
 public:
  PixelBarrelLayerBuilder(){};
  PixelBarrelLayer* build(const GeometricDet* aPixelBarrelLayer,
			  const TrackerGeometry* theGeomDetGeometry);

  
};


#endif 
