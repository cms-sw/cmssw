#ifndef TkDetLayers_Phase2EndcapRingBuilder_h
#define TkDetLayers_Phase2EndcapRingBuilder_h


#include "Phase2EndcapRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2EndcapRing 
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapRingBuilder {  
 public:
  Phase2EndcapRingBuilder(){};
  Phase2EndcapRing* build(const GeometricDet* aPhase2EndcapRing,
			    const TrackerGeometry* theGeomDetGeometry,
			    const bool useBrothers = true) __attribute__ ((cold));

  
};


#pragma GCC visibility pop
#endif 
