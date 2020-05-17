#include "TECPetalBuilder.h"
#include "TECWedgeBuilder.h"
#include "CompositeTECPetal.h"

using namespace edm;
using namespace std;

TECPetal* TECPetalBuilder::build(const GeometricDet* aTECPetal, const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*> theGeometricWedges = aTECPetal->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricWedges.size(): " << theGeometricWedges.size() ;

  vector<const TECWedge*> theInnerWedges;
  vector<const TECWedge*> theOuterWedges;

  double meanZ = (theGeometricWedges[0]->positionBounds().z() + theGeometricWedges[1]->positionBounds().z()) / 2;

  TECWedgeBuilder myWedgeBuilder;

  for (auto theGeometricWedge : theGeometricWedges) {
    if (std::abs(theGeometricWedge->positionBounds().z()) < std::abs(meanZ))
      theInnerWedges.push_back(myWedgeBuilder.build(theGeometricWedge, theGeomDetGeometry));

    if (std::abs(theGeometricWedge->positionBounds().z()) > std::abs(meanZ))
      theOuterWedges.push_back(myWedgeBuilder.build(theGeometricWedge, theGeomDetGeometry));
  }

  //edm::LogInfo(TkDetLayers) << "theInnerWededges.size(): " << theInnerWedges.size() ;
  //edm::LogInfo(TkDetLayers) << "theOuterWededges.size(): " << theOuterWedges.size() ;

  return new CompositeTECPetal(theInnerWedges, theOuterWedges);
}
