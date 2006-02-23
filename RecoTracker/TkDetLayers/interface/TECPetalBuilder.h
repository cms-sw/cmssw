#ifndef TkDetLayers_TECPetalBuilder_h
#define TkDetLayers_TECPetalBuilder_h


#include "RecoTracker/TkDetLayers/interface/TECPetal.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TECPetal 
 */
using namespace edm;
using namespace std;

class TECPetalBuilder {  
 public:
  TECPetalBuilder(){};
  TECPetal* build(const GeometricDet* aTECPetal,
	      const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
