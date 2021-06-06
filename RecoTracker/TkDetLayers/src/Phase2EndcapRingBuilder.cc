#include "Phase2EndcapRingBuilder.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

Phase2EndcapRing* Phase2EndcapRingBuilder::build(const GeometricDet* aPhase2EndcapRing,
                                                 const TrackerGeometry* theGeomDetGeometry,
                                                 const bool useBrothers) {
  vector<const GeometricDet*> allGeometricDets = aPhase2EndcapRing->components();
  vector<const GeometricDet*> compGeometricDets;
  LogDebug("TkDetLayers") << "Phase2EndcapRingBuilder with #Modules: " << allGeometricDets.size() << std::endl;

  vector<const GeomDet*> frontGeomDets;
  vector<const GeomDet*> backGeomDets;
  double meanZ = 0;

  if (!useBrothers) {
    //---- to evaluate meanZ
    for (vector<const GeometricDet*>::const_iterator compGeometricDets = allGeometricDets.begin();
         compGeometricDets != allGeometricDets.end();
         compGeometricDets++) {
      LogTrace("TkDetLayers") << " compGeometricDets->positionBounds().perp() "
                              << (*compGeometricDets)->positionBounds().z() << std::endl;
      meanZ = meanZ + (*compGeometricDets)->positionBounds().z();
    }
    meanZ = meanZ / allGeometricDets.size();
    LogDebug("TkDetLayers") << " meanZ " << meanZ << std::endl;

    for (vector<const GeometricDet*>::const_iterator compGeometricDets = allGeometricDets.begin();
         compGeometricDets != allGeometricDets.end();
         compGeometricDets++) {
      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet((*compGeometricDets)->geographicalId());

      if (fabs((*compGeometricDets)->positionBounds().z()) < fabs(meanZ))
        frontGeomDets.push_back(theGeomDet);

      if (fabs((*compGeometricDets)->positionBounds().z()) > fabs(meanZ))
        backGeomDets.push_back(theGeomDet);

      if (fabs((*compGeometricDets)->positionBounds().z()) == fabs(meanZ))
        throw DetLayerException("Not possible to assiciate this GeometricDet in front or back");
    }

    LogDebug("TkDetLayers") << "frontGeomDets.size(): " << frontGeomDets.size();
    LogDebug("TkDetLayers") << "backGeomDets.size(): " << backGeomDets.size();

    return new Phase2EndcapRing(frontGeomDets, backGeomDets);

  } else {
    vector<const GeomDet*> frontGeomDetBrothers;
    vector<const GeomDet*> backGeomDetBrothers;
    vector<const GeometricDet*> compGeometricDets;

    //---- to evaluate meanZ
    double meanZ = 0;
    double meanZBrothers = 0;
    for (vector<const GeometricDet*>::const_iterator it = allGeometricDets.begin(); it != allGeometricDets.end();
         it++) {
      compGeometricDets = (*it)->components();
      if (compGeometricDets.size() != 2) {
        throw DetLayerException("Phase2OTEndcapRing is considered as a stack but does not have two components");
      } else {
        LogTrace("TkDetLayers") << " compGeometricDets[0]->positionBounds().perp() "
                                << compGeometricDets[0]->positionBounds().z() << std::endl;
        LogTrace("TkDetLayers") << " compGeometricDets[1]->positionBounds().perp() "
                                << compGeometricDets[1]->positionBounds().z() << std::endl;
        meanZ = meanZ + compGeometricDets[0]->positionBounds().z();
        meanZBrothers = meanZBrothers + compGeometricDets[1]->positionBounds().z();
      }
    }
    meanZ = meanZ / allGeometricDets.size();
    meanZBrothers = meanZBrothers / allGeometricDets.size();
    LogDebug("TkDetLayers") << " meanZ Lower " << meanZ << std::endl;
    LogDebug("TkDetLayers") << " meanZ Upper " << meanZBrothers << std::endl;

    for (vector<const GeometricDet*>::const_iterator it = allGeometricDets.begin(); it != allGeometricDets.end();
         it++) {
      compGeometricDets = (*it)->components();
      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(compGeometricDets[0]->geographicalId());

      if (fabs(compGeometricDets[0]->positionBounds().z()) < fabs(meanZ))
        frontGeomDets.push_back(theGeomDet);

      if (fabs(compGeometricDets[0]->positionBounds().z()) > fabs(meanZ))
        backGeomDets.push_back(theGeomDet);

      const GeomDet* theGeomDetBrother = theGeomDetGeometry->idToDet(compGeometricDets[1]->geographicalId());

      if (fabs(compGeometricDets[1]->positionBounds().z()) < fabs(meanZBrothers))
        frontGeomDetBrothers.push_back(theGeomDetBrother);

      if (fabs(compGeometricDets[1]->positionBounds().z()) > fabs(meanZBrothers))
        backGeomDetBrothers.push_back(theGeomDetBrother);

      if (fabs(compGeometricDets[0]->positionBounds().z()) == fabs(meanZ) ||
          fabs(compGeometricDets[1]->positionBounds().z()) == fabs(meanZBrothers))
        throw DetLayerException("Not possible to assiciate components of this GeometricDet in front or back");
    }

    LogDebug("TkDetLayers") << "frontGeomDets.size(): " << frontGeomDets.size();
    LogDebug("TkDetLayers") << "backGeomDets.size(): " << backGeomDets.size();
    LogDebug("TkDetLayers") << "frontGeomDetsBro.size(): " << frontGeomDetBrothers.size();
    LogDebug("TkDetLayers") << "backGeomDetsBro.size(): " << backGeomDetBrothers.size();

    return new Phase2EndcapRing(frontGeomDets, backGeomDets, frontGeomDetBrothers, backGeomDetBrothers);
  }
  return new Phase2EndcapRing(frontGeomDets, backGeomDets);
}
