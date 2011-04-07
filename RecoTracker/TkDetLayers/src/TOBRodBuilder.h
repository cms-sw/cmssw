#ifndef TkDetLayers_TOBRodBuilder_h
#define TkDetLayers_TOBRodBuilder_h


#include "TOBRod.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TOBRod 
 */

class TOBRodBuilder {  
 public:
  TOBRodBuilder(){};
  TOBRod* build(const GeometricDet* negTOBRod,
		const GeometricDet* posTOBRod,
		const TrackerGeometry* theGeomDetGeometry);

  
};


#endif 
