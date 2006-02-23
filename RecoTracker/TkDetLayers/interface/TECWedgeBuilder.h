#ifndef TkDetLayers_TECWedgeBuilder_h
#define TkDetLayers_TECWedgeBuilder_h


#include "RecoTracker/TkDetLayers/interface/TECWedge.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TECWedge 
 */
using namespace edm;
using namespace std;

class TECWedgeBuilder {  
 public:
  TECWedgeBuilder(){};
  TECWedge* build(const GeometricDet* aTECWedge,
	      const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
