#include "PixelRodBuilder.h"

using namespace edm;
using namespace std;

PixelRod* PixelRodBuilder::build(const GeometricDet* aRod, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> allGeometricDets = aRod->components();

  vector<const GeomDet*> theGeomDets;
  for (auto& allGeometricDet : allGeometricDets) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(allGeometricDet->geographicalID());
    theGeomDets.push_back(theGeomDet);
  }

  return new PixelRod(theGeomDets);
}
