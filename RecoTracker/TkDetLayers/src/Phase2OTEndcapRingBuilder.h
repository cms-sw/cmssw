#ifndef TkDetLayers_Phase2OTEndcapRingBuilder_h
#define TkDetLayers_Phase2OTEndcapRingBuilder_h


#include "Phase2OTEndcapRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2OTEndcapRing 
 */

#pragma GCC visibility push(hidden)
class Phase2OTEndcapRingBuilder {  
 public:
  Phase2OTEndcapRingBuilder(){};
  Phase2OTEndcapRing* build(const GeometricDet* aPhase2OTEndcapRing,
			const TrackerGeometry* theGeomDetGeometry) __attribute__ ((cold));

  
};


#pragma GCC visibility pop
#endif 
