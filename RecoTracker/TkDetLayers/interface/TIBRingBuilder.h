#ifndef TkDetLayers_TIBRingBuilder_h
#define TkDetLayers_TIBRingBuilder_h


#include "RecoTracker/TkDetLayers/interface/TIBRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TIBRing 
 */
using namespace edm;
using namespace std;

class TIBRingBuilder {  
 public:
  TIBRingBuilder(){};
  TIBRing* build(const vector<const GeometricDet*>& detsInRing,
		 const TrackerGeometry* theGeomDetGeometry);

  
};


#endif 
