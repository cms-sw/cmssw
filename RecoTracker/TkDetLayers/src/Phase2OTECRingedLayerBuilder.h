#ifndef TkDetLayers_Phase2OTECRingedLayerBuilder_h
#define TkDetLayers_Phase2OTECRingedLayerBuilder_h


#include "Phase2OTECRingedLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2OTECRingedLayer 
 */

#pragma GCC visibility push(hidden)
class Phase2OTECRingedLayerBuilder {  
 public:
  Phase2OTECRingedLayerBuilder(){};
  Phase2OTECRingedLayer* build(const GeometricDet* aPhase2OTECRingedLayer,
			       const TrackerGeometry* theGeomDetGeometry) __attribute__ ((cold));

   
};


#pragma GCC visibility pop
#endif 
