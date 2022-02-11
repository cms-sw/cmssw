#include "Phase2EndcapLayerDoubleDiskBuilder.h"
#include "Phase2EndcapSubDiskBuilder.h"

using namespace edm;
using namespace std;

Phase2EndcapLayerDoubleDisk* Phase2EndcapLayerDoubleDiskBuilder::build(const GeometricDet* aPhase2EndcapLayerDoubleDisk,
                                                                       const TrackerGeometry* theGeomDetGeometry) {
  LogTrace("TkDetLayers") << "Phase2EndcapLayerDoubleDiskBuilder::build";
  const auto& theSubDisks = aPhase2EndcapLayerDoubleDisk->components();
  LogTrace("TkDetLayers") << "theSubDisks.size(): " << theSubDisks.size();

  Phase2EndcapSubDiskBuilder myBuilder;
  vector<const Phase2EndcapSubDisk*> thePhase2EndcapSubDisks;
  thePhase2EndcapSubDisks.reserve(theSubDisks.size());

  for (vector<const GeometricDet*>::const_iterator it = theSubDisks.begin(); it != theSubDisks.end(); it++) {
    thePhase2EndcapSubDisks.push_back(myBuilder.build(*it, theGeomDetGeometry));
  }

  return new Phase2EndcapLayerDoubleDisk(thePhase2EndcapSubDisks);
}
