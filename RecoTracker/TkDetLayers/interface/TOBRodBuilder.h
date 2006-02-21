#ifndef TkDetLayers_TOBRodBuilder_h
#define TkDetLayers_TOBRodBuilder_h


#include "RecoTracker/TkDetLayers/interface/TOBRod.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TOBRod 
 */
using namespace edm;
using namespace std;

class TOBRodBuilder {  
 public:
  TOBRodBuilder(){};
  TOBRod* build(const GeometricDet* aTOBRod,
	      ESHandle<TrackingGeometry> pTrackingGeometry);

  
};


#endif 
