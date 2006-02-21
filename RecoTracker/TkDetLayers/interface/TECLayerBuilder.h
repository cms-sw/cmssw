#ifndef TkDetLayers_TECLayerBuilder_h
#define TkDetLayers_TECLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/TECLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TECLayer 
 */
using namespace edm;
using namespace std;

class TECLayerBuilder {  
 public:
  TECLayerBuilder(){};
  TECLayer* build(const GeometricDet* aTECLayer,
	      ESHandle<TrackingGeometry> pTrackingGeometry);

  
};


#endif 
