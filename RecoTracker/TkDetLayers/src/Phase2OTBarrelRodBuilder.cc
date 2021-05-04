#include "Phase2OTBarrelRodBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

Phase2OTBarrelRod* Phase2OTBarrelRodBuilder::build(const GeometricDet* thePhase2OTBarrelRod,
                                                   const TrackerGeometry* theGeomDetGeometry,
                                                   const bool useBrothers) {
  vector<const GeometricDet*> allGeometricDets = thePhase2OTBarrelRod->components();
  LogDebug("TkDetLayers") << "Phase2OTBarrelRodBuilder with #Modules: " << allGeometricDets.size() << std::endl;
  LogDebug("TkDetLayers") << "                           useBrothers: " << useBrothers << std::endl;

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;
  vector<const GeomDet*> innerGeomDetBrothers;
  vector<const GeomDet*> outerGeomDetBrothers;

  double meanR = 0;

  if (!useBrothers) {
    for (auto const& compGeometricDets : allGeometricDets) {
      meanR = meanR + compGeometricDets->positionBounds().perp();
    }
    meanR = meanR / allGeometricDets.size();
    LogDebug("TkDetLayers") << " meanR Lower " << meanR << std::endl;
    for (auto const& compGeometricDets : allGeometricDets) {
      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(compGeometricDets->geographicalId());

      if (compGeometricDets->positionBounds().perp() < meanR)
        innerGeomDets.push_back(theGeomDet);

      if (compGeometricDets->positionBounds().perp() > meanR)
        outerGeomDets.push_back(theGeomDet);
    }

    LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size();
    LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size();

  } else {
    vector<const GeometricDet*> compGeometricDets;

    double meanRBrothers = 0;
    for (auto& it : allGeometricDets) {
      compGeometricDets = it->components();
      if (compGeometricDets.size() != 2) {
        LogDebug("TkDetLayers") << " Stack not with two components but with " << compGeometricDets.size() << std::endl;
      } else {
        meanR = meanR + compGeometricDets[0]->positionBounds().perp();
        meanRBrothers = meanRBrothers + compGeometricDets[1]->positionBounds().perp();
      }
    }
    meanR = meanR / allGeometricDets.size();
    meanRBrothers = meanRBrothers / allGeometricDets.size();
    LogDebug("TkDetLayers") << " meanR Lower " << meanR << std::endl;
    LogDebug("TkDetLayers") << " meanR Upper " << meanRBrothers << std::endl;

    for (auto& it : allGeometricDets) {
      compGeometricDets = it->components();
      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(compGeometricDets[0]->geographicalId());
      LogTrace("TkDetLayers") << " inserting " << compGeometricDets[0]->geographicalId().rawId() << std::endl;

      if (compGeometricDets[0]->positionBounds().perp() < meanR)
        innerGeomDets.push_back(theGeomDet);

      else
        outerGeomDets.push_back(theGeomDet);

      const GeomDet* theGeomDetBrother = theGeomDetGeometry->idToDet(compGeometricDets[1]->geographicalId());
      LogTrace("TkDetLayers") << " inserting " << compGeometricDets[1]->geographicalId().rawId() << std::endl;
      if (compGeometricDets[1]->positionBounds().perp() < meanRBrothers)
        innerGeomDetBrothers.push_back(theGeomDetBrother);

      else
        outerGeomDetBrothers.push_back(theGeomDetBrother);
    }

    LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size();
    LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size();
    LogDebug("TkDetLayers") << "innerGeomDetsBro.size(): " << innerGeomDetBrothers.size();
    LogDebug("TkDetLayers") << "outerGeomDetsBro.size(): " << outerGeomDetBrothers.size();
  }

  return new Phase2OTBarrelRod(innerGeomDets, outerGeomDets, innerGeomDetBrothers, outerGeomDetBrothers);
}
