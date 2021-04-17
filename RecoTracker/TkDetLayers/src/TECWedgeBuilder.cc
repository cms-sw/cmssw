#include "TECWedgeBuilder.h"
#include "CompositeTECWedge.h"
#include "SimpleTECWedge.h"

using namespace edm;
using namespace std;

TECWedge* TECWedgeBuilder::build(const GeometricDet* aTECWedge, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> theGeometricDets = aTECWedge->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricDets.size(): " << theGeometricDets.size() ;

  if (theGeometricDets.size() == 1) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(theGeometricDets.front()->geographicalId());
    return new SimpleTECWedge(theGeomDet);
  }

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;

  //---- to evaluate meanZ
  double meanZ = 0;
  for (vector<const GeometricDet*>::const_iterator it = theGeometricDets.begin(); it != theGeometricDets.end(); it++) {
    meanZ = meanZ + (*it)->positionBounds().z();
  }

  meanZ = meanZ / theGeometricDets.size();
  //edm::LogInfo(TkDetLayers) << "meanZ: " << meanZ ;
  //----

  for (vector<const GeometricDet*>::const_iterator it = theGeometricDets.begin(); it != theGeometricDets.end(); it++) {
    //double theGeometricDetRposition = (*it)->positionBounds().perp();
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet((*it)->geographicalId());
    //double theGeomDetRposition = theGeomDet->surface().position().perp();

    if (std::abs((*it)->positionBounds().z()) < std::abs(meanZ))
      innerGeomDets.push_back(theGeomDet);

    if (std::abs((*it)->positionBounds().z()) > std::abs(meanZ))
      outerGeomDets.push_back(theGeomDet);
  }

  //edm::LogInfo(TkDetLayers) << "innerGeomDets.size(): " << innerGeomDets.size() ;
  //edm::LogInfo(TkDetLayers) << "outerGeomDets.size(): " << outerGeomDets.size() ;

  return new CompositeTECWedge(innerGeomDets, outerGeomDets);
}
