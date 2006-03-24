#ifndef TkDetLayers_TIBLayerBuilder_h
#define TkDetLayers_TIBLayerBuilder_h


#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for TIBLayer 
 */
using namespace edm;
using namespace std;

class TIBLayerBuilder {  
 public:
  TIBLayerBuilder(){};
  TIBLayer* build(const GeometricDet* aTIBLayer,
		  const TrackingGeometry* theGeomDetGeometry);

  void constructRings(vector<const GeometricDet*>& theGeometricRods,
		      vector<vector<const GeometricDet*> >& innerGeometricDetRings,
		      vector<vector<const GeometricDet*> >& outerGeometricDetRings);
  
};


#endif 
