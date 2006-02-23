#ifndef TkDetLayers_TOBLayerBuilder_h
#define TkDetLayers_TOBLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TOBLayer 
 */
using namespace edm;
using namespace std;

class TOBLayerBuilder {  
 public:
  TOBLayerBuilder(){};
  TOBLayer* build(const GeometricDet* aTOBLayer,
		  const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
