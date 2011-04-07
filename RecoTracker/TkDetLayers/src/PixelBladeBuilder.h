#ifndef TkDetLayers_PixelBladeBuilder_h
#define TkDetLayers_PixelBladeBuilder_h


#include "PixelBlade.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for PixelBlade 
 */

#pragma GCC visibility push(hidden)
class PixelBladeBuilder {  
 public:
  PixelBladeBuilder(){};
  PixelBlade* build(const GeometricDet* geometricDetFrontPanel,
		    const GeometricDet* geometricDetBackPanel,
		    const TrackerGeometry* theGeomDetGeometry);
  
};


#pragma GCC visibility pop
#endif 
