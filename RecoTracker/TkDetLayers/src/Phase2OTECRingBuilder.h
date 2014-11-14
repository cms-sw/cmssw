#ifndef TkDetLayers_Phase2OTECRingBuilder_h
#define TkDetLayers_Phase2OTECRingBuilder_h


#include "Phase2OTECRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2OTECRing 
 */

#pragma GCC visibility push(hidden)
class Phase2OTECRingBuilder {  
 public:
  Phase2OTECRingBuilder(){};
  Phase2OTECRing* build(const GeometricDet* aPhase2OTECRing,
			const TrackerGeometry* theGeomDetGeometry) __attribute__ ((cold));

  
};


#pragma GCC visibility pop
#endif 
