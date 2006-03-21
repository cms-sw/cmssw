#ifndef TkDetLayers_TECWedgeBuilder_h
#define TkDetLayers_TECWedgeBuilder_h


#include "RecoTracker/TkDetLayers/interface/TECWedge.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TECWedge 
 */
using namespace edm;
using namespace std;

class TECWedgeBuilder {  
 public:
  TECWedgeBuilder(){};
  TECWedge* build(const GeometricDet* aTECWedge,
	      const TrackerGeometry* theGeomDetGeometry);

  
};


#endif 
