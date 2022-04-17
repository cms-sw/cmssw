#include "Phase2EndcapSubDiskBuilder.h"
#include "Phase2EndcapSingleRingBuilder.h"

using namespace edm;
using namespace std;

Phase2EndcapSubDisk* Phase2EndcapSubDiskBuilder::build(const GeometricDet* aPhase2EndcapSubDisk,
                                                       const TrackerGeometry* theGeomDetGeometry) {
  LogTrace("TkDetLayers") << "Phase2EndcapSubDiskBuilder::build";
  vector<const GeometricDet*> theGeometricRings = aPhase2EndcapSubDisk->components();
  LogTrace("TkDetLayers") << "theGeometricRings.size(): " << theGeometricRings.size();

  Phase2EndcapSingleRingBuilder myBuilder;
  vector<const Phase2EndcapSingleRing*> thePhase2EndcapSingleRings;

  for (vector<const GeometricDet*>::const_iterator it = theGeometricRings.begin(); it != theGeometricRings.end();
       it++) {
    // if we are in the phaseII OT, it will use the brothers to build pt modules
    // if we are in the phaseII pixel detector, it will not
    thePhase2EndcapSingleRings.push_back(myBuilder.build(*it, theGeomDetGeometry));
  }

  return new Phase2EndcapSubDisk(thePhase2EndcapSingleRings);
}
