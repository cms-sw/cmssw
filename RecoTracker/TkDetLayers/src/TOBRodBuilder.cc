#include "TOBRodBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

TOBRod* TOBRodBuilder::build(const GeometricDet* negTOBRod,
                             const GeometricDet* posTOBRod,
                             const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> theNegativeGeometricDets;
  if (negTOBRod != nullptr)
    theNegativeGeometricDets = negTOBRod->components();
  vector<const GeometricDet*> thePositiveGeometricDets;
  if (posTOBRod != nullptr)
    thePositiveGeometricDets = posTOBRod->components();

  vector<const GeometricDet*> allGeometricDets = theNegativeGeometricDets;
  allGeometricDets.insert(allGeometricDets.end(), thePositiveGeometricDets.begin(), thePositiveGeometricDets.end());

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;

  double meanR = (allGeometricDets[0]->positionBounds().perp() + allGeometricDets[1]->positionBounds().perp()) / 2;
  for (vector<const GeometricDet*>::iterator it = allGeometricDets.begin(); it != allGeometricDets.end(); it++) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet((*it)->geographicalId());

    if ((*it)->positionBounds().perp() < meanR)
      innerGeomDets.push_back(theGeomDet);

    if ((*it)->positionBounds().perp() > meanR)
      outerGeomDets.push_back(theGeomDet);
  }

  //LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size() ;
  //LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size() ;
  return new TOBRod(innerGeomDets, outerGeomDets);
}
