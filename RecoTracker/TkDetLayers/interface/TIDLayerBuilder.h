#ifndef TkDetLayers_TIDLayerBuilder_h
#define TkDetLayers_TIDLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/TIDLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TIDLayer 
 */
using namespace edm;
using namespace std;

class TIDLayerBuilder {  
 public:
  TIDLayerBuilder(){};
  TIDLayer* build(const GeometricDet* aTIDLayer,
	      ESHandle<TrackingGeometry> pTrackingGeometry);

  
};


#endif 
