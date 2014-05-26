#ifndef TkDetLayers_Phase2OTECLayerBuilder_h
#define TkDetLayers_Phase2OTECLayerBuilder_h


#include "Phase2OTECLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2OTECLayer 
 */

#pragma GCC visibility push(hidden)
class Phase2OTECLayerBuilder {  
 public:
  Phase2OTECLayerBuilder(){};
  ForwardDetLayer* build(const GeometricDet* aPhase2OTECLayer,
			   const TrackerGeometry* theGeomDetGeometry);

  
};


#pragma GCC visibility pop
#endif 
