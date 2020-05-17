#include "TIDLayerBuilder.h"
#include "TIDRingBuilder.h"

using namespace edm;
using namespace std;

TIDLayer* TIDLayerBuilder::build(const GeometricDet* aTIDLayer, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> theGeometricRings = aTIDLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricRings.size(): " << theGeometricRings.size() ;

  TIDRingBuilder myBuilder;
  vector<const TIDRing*> theTIDRings;

  for (auto theGeometricRing : theGeometricRings) {
    theTIDRings.push_back(myBuilder.build(theGeometricRing, theGeomDetGeometry));
  }

  return new TIDLayer(theTIDRings);
}
