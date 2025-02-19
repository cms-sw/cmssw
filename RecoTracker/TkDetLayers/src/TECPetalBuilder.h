#ifndef TkDetLayers_TECPetalBuilder_h
#define TkDetLayers_TECPetalBuilder_h


#include "TECPetal.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TECPetal 
 */

#pragma GCC visibility push(hidden)
class TECPetalBuilder {  
 public:
  TECPetalBuilder(){};
  TECPetal* build(const GeometricDet* aTECPetal,
		  const TrackerGeometry* theGeomDetGeometry);

  
};


#pragma GCC visibility pop
#endif 
