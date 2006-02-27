#ifndef TkDetLayers_PixelBarrelLayerBuilder_h
#define TkDetLayers_PixelBarrelLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for PixelBarrelLayer 
 */
using namespace edm;
using namespace std;

class PixelBarrelLayerBuilder {  
 public:
  PixelBarrelLayerBuilder(){};
  PixelBarrelLayer* build(const GeometricDet* aPixelBarrelLayer,
			  const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
