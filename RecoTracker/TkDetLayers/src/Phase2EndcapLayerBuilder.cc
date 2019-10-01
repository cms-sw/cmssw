#include "Phase2EndcapLayerBuilder.h"
#include "Phase2EndcapRingBuilder.h"

using namespace edm;
using namespace std;

Phase2EndcapLayer* Phase2EndcapLayerBuilder::build(const GeometricDet* aPhase2EndcapLayer,
                                                   const TrackerGeometry* theGeomDetGeometry,
                                                   const bool useBrothers) {
  LogTrace("TkDetLayers") << "Phase2EndcapLayerBuilder::build";
  vector<const GeometricDet*> theGeometricRings = aPhase2EndcapLayer->components();
  LogTrace("TkDetLayers") << "theGeometricRings.size(): " << theGeometricRings.size();

  Phase2EndcapRingBuilder myBuilder;
  vector<const Phase2EndcapRing*> thePhase2EndcapRings;

  for (vector<const GeometricDet*>::const_iterator it = theGeometricRings.begin(); it != theGeometricRings.end();
       it++) {
    // if we are in the phaseII OT, it will use the brothers to build pt modules
    // if we are in the phaseII pixel detector, it will not
    thePhase2EndcapRings.push_back(myBuilder.build(*it, theGeomDetGeometry, useBrothers));
  }

  return new Phase2EndcapLayer(thePhase2EndcapRings, useBrothers);
}
