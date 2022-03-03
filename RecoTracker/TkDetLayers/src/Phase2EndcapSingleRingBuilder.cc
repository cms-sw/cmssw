#include "Phase2EndcapSingleRingBuilder.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

Phase2EndcapSingleRing* Phase2EndcapSingleRingBuilder::build(const GeometricDet* aPhase2EndcapSingleRing,
                                                             const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> allGeometricDets = aPhase2EndcapSingleRing->components();
  vector<const GeometricDet*> compGeometricDets;
  vector<const GeomDet*> allDets;

  for (vector<const GeometricDet*>::const_iterator compGeometricDets = allGeometricDets.begin();
       compGeometricDets != allGeometricDets.end();
       compGeometricDets++) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet((*compGeometricDets)->geographicalId());
    allDets.push_back(theGeomDet);
  }

  LogDebug("TkDetLayers") << "Phase2EndcapSingleRingBuilder with #Modules: " << allGeometricDets.size() << std::endl;

  return new Phase2EndcapSingleRing(allDets);
}
