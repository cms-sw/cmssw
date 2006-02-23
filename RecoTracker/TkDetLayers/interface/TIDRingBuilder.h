#ifndef TkDetLayers_TIDRingBuilder_h
#define TkDetLayers_TIDRingBuilder_h


#include "RecoTracker/TkDetLayers/interface/TIDRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TIDRing 
 */
using namespace edm;
using namespace std;

class TIDRingBuilder {  
 public:
  TIDRingBuilder(){};
  TIDRing* build(const GeometricDet* aTIDRing,
	      const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
