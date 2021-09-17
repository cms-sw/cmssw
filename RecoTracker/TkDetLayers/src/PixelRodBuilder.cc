#include "PixelRodBuilder.h"

using namespace edm;
using namespace std;

PixelRod* PixelRodBuilder::build(const GeometricDet* aRod, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> allGeometricDets = aRod->components();

  vector<const GeomDet*> theGeomDets;
  for (vector<const GeometricDet*>::iterator it = allGeometricDets.begin(); it != allGeometricDets.end(); it++) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet((*it)->geographicalId());
    theGeomDets.push_back(theGeomDet);
  }

  return new PixelRod(theGeomDets);
}
