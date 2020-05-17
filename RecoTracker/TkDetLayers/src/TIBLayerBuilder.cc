#include "TIBLayerBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TIBRingBuilder.h"

using namespace edm;
using namespace std;

TIBLayer* TIBLayerBuilder::build(const GeometricDet* aTIBLayer, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> theGeometricRods = aTIBLayer->components();

  vector<vector<const GeometricDet*> > innerGeometricDetRings;
  vector<vector<const GeometricDet*> > outerGeometricDetRings;

  constructRings(theGeometricRods, innerGeometricDetRings, outerGeometricDetRings);

  TIBRingBuilder myRingBuilder;

  vector<const TIBRing*> innerRings;
  vector<const TIBRing*> outerRings;

  for (unsigned int i = 0; i < innerGeometricDetRings.size(); i++) {
    innerRings.push_back(myRingBuilder.build(innerGeometricDetRings[i], theGeomDetGeometry));
    outerRings.push_back(myRingBuilder.build(outerGeometricDetRings[i], theGeomDetGeometry));
  }

  return new TIBLayer(innerRings, outerRings);
}

void TIBLayerBuilder::constructRings(vector<const GeometricDet*>& theGeometricRods,
                                     vector<vector<const GeometricDet*> >& innerGeometricDetRings,
                                     vector<vector<const GeometricDet*> >& outerGeometricDetRings) {
  double meanPerp = 0;
  for (auto theGeometricRod : theGeometricRods) {
    meanPerp = meanPerp + theGeometricRod->positionBounds().perp();
  }
  meanPerp = meanPerp / theGeometricRods.size();

  vector<const GeometricDet*> theInnerGeometricRods;
  vector<const GeometricDet*> theOuterGeometricRods;

  for (auto theGeometricRod : theGeometricRods) {
    if (theGeometricRod->positionBounds().perp() < meanPerp)
      theInnerGeometricRods.push_back(theGeometricRod);
    if (theGeometricRod->positionBounds().perp() > meanPerp)
      theOuterGeometricRods.push_back(theGeometricRod);
  }

  size_t innerLeftRodMaxSize = 0;
  size_t innerRightRodMaxSize = 0;
  size_t outerLeftRodMaxSize = 0;
  size_t outerRightRodMaxSize = 0;

  for (auto theInnerGeometricRod : theInnerGeometricRods) {
    if (theInnerGeometricRod->positionBounds().z() < 0)
      innerLeftRodMaxSize = max(innerLeftRodMaxSize, (*theInnerGeometricRod).components().size());
    if (theInnerGeometricRod->positionBounds().z() > 0)
      innerRightRodMaxSize = max(innerRightRodMaxSize, (*theInnerGeometricRod).components().size());
  }

  for (auto theOuterGeometricRod : theOuterGeometricRods) {
    if (theOuterGeometricRod->positionBounds().z() < 0)
      outerLeftRodMaxSize = max(outerLeftRodMaxSize, (*theOuterGeometricRod).components().size());
    if (theOuterGeometricRod->positionBounds().z() > 0)
      outerRightRodMaxSize = max(outerRightRodMaxSize, (*theOuterGeometricRod).components().size());
  }

  LogDebug("TkDetLayers") << "innerLeftRodMaxSize: " << innerLeftRodMaxSize;
  LogDebug("TkDetLayers") << "innerRightRodMaxSize: " << innerRightRodMaxSize;

  LogDebug("TkDetLayers") << "outerLeftRodMaxSize: " << outerLeftRodMaxSize;
  LogDebug("TkDetLayers") << "outerRightRodMaxSize: " << outerRightRodMaxSize;

  for (unsigned int i = 0; i < (innerLeftRodMaxSize + innerRightRodMaxSize); i++) {
    innerGeometricDetRings.emplace_back();
  }

  for (unsigned int i = 0; i < (outerLeftRodMaxSize + outerRightRodMaxSize); i++) {
    outerGeometricDetRings.emplace_back();
  }

  for (unsigned int ringN = 0; ringN < innerLeftRodMaxSize; ringN++) {
    for (auto theInnerGeometricRod : theInnerGeometricRods) {
      if (theInnerGeometricRod->positionBounds().z() < 0) {
        if ((*theInnerGeometricRod).components().size() > ringN)
          innerGeometricDetRings[ringN].push_back((*theInnerGeometricRod).components()[ringN]);
      }
    }
  }

  for (unsigned int ringN = 0; ringN < innerRightRodMaxSize; ringN++) {
    for (auto theInnerGeometricRod : theInnerGeometricRods) {
      if (theInnerGeometricRod->positionBounds().z() > 0) {
        if ((*theInnerGeometricRod).components().size() > ringN)
          innerGeometricDetRings[innerLeftRodMaxSize + ringN].push_back((*theInnerGeometricRod).components()[ringN]);
      }
    }
  }

  for (unsigned int ringN = 0; ringN < outerLeftRodMaxSize; ringN++) {
    for (auto theOuterGeometricRod : theOuterGeometricRods) {
      if (theOuterGeometricRod->positionBounds().z() < 0) {
        if ((*theOuterGeometricRod).components().size() > ringN)
          outerGeometricDetRings[ringN].push_back((*theOuterGeometricRod).components()[ringN]);
      }
    }
  }

  for (unsigned int ringN = 0; ringN < outerRightRodMaxSize; ringN++) {
    for (auto theOuterGeometricRod : theOuterGeometricRods) {
      if (theOuterGeometricRod->positionBounds().z() > 0) {
        if ((*theOuterGeometricRod).components().size() > ringN)
          outerGeometricDetRings[outerLeftRodMaxSize + ringN].push_back((*theOuterGeometricRod).components()[ringN]);
      }
    }
  }
}
