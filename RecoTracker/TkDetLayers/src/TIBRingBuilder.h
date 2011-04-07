#ifndef TkDetLayers_TIBRingBuilder_h
#define TkDetLayers_TIBRingBuilder_h


#include "TIBRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TIBRing 
 */

class TIBRingBuilder {  
 public:
  TIBRingBuilder(){};
  TIBRing* build(const std::vector<const GeometricDet*>& detsInRing,
		 const TrackerGeometry* theGeomDetGeometry);

  
};


#endif 
