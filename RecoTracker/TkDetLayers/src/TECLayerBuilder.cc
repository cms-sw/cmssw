#include "TECLayerBuilder.h"
#include "TECPetalBuilder.h"

using namespace edm;
using namespace std;

TECLayer* TECLayerBuilder::build(const GeometricDet* aTECLayer, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> theGeometricDetPetals = aTECLayer->components();
  vector<const TECPetal*> theInnerPetals;
  vector<const TECPetal*> theOuterPetals;

  //edm::LogInfo(TkDetLayers) << "theGeometricDetPetals.size(): " << theGeometricDetPetals.size() ;

  double meanZ =
      (theGeometricDetPetals.front()->positionBounds().z() + theGeometricDetPetals.back()->positionBounds().z()) / 2;

  TECPetalBuilder myPetalBuilder;

  for (auto theGeometricDetPetal : theGeometricDetPetals) {
    if (std::abs(theGeometricDetPetal->positionBounds().z()) < std::abs(meanZ))
      theInnerPetals.push_back(myPetalBuilder.build(theGeometricDetPetal, theGeomDetGeometry));

    if (std::abs(theGeometricDetPetal->positionBounds().z()) > std::abs(meanZ))
      theOuterPetals.push_back(myPetalBuilder.build(theGeometricDetPetal, theGeomDetGeometry));
  }

  return new TECLayer(theInnerPetals, theOuterPetals);
}
