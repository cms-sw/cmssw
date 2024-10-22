#include "PixelRodBuilder.h"

using namespace edm;
using namespace std;

PixelRod* PixelRodBuilder::build(const GeometricDet* aRod, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> allGeometricDets = aRod->components();

  vector<const GeomDet*> theGeomDets;
  vector<const GeometricDet*> compGeometricDets;
  for (auto& it : allGeometricDets) {
    compGeometricDets = it->components();
    if (it->type() == GeometricDet::ITPhase2Combined) {
      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(compGeometricDets[0]->geographicalId());
      theGeomDets.push_back(theGeomDet);
      const GeomDet* theGeomDetBrother = theGeomDetGeometry->idToDet(compGeometricDets[1]->geographicalId());
      theGeomDets.push_back(theGeomDetBrother);
    } else if (it->type() == GeometricDet::DetUnit) {
      const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(it->geographicalId());
      theGeomDets.push_back(theGeomDet);
    }
  }
  return new PixelRod(theGeomDets);
}
