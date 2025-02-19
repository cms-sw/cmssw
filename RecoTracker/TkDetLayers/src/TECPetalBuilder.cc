#include "TECPetalBuilder.h"
#include "TECWedgeBuilder.h"
#include "CompositeTECPetal.h"

using namespace edm;
using namespace std;

TECPetal* TECPetalBuilder::build(const GeometricDet* aTECPetal,
				 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricWedges = aTECPetal->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricWedges.size(): " << theGeometricWedges.size() ;
  
  vector<const TECWedge*> theInnerWedges;
  vector<const TECWedge*> theOuterWedges;

  double meanZ = ( theGeometricWedges[0]->positionBounds().z() + 
		   theGeometricWedges[1]->positionBounds().z() )/2;

  TECWedgeBuilder myWedgeBuilder;
   

  for(vector<const GeometricDet*>::const_iterator it=theGeometricWedges.begin();
      it!=theGeometricWedges.end();it++){
    if( fabs((*it)->positionBounds().z()) < fabs(meanZ) ) 
      theInnerWedges.push_back(myWedgeBuilder.build(*it,theGeomDetGeometry));
    
    if( fabs((*it)->positionBounds().z()) > fabs(meanZ) ) 
      theOuterWedges.push_back(myWedgeBuilder.build(*it,theGeomDetGeometry));
  }
  
  //edm::LogInfo(TkDetLayers) << "theInnerWededges.size(): " << theInnerWedges.size() ;
  //edm::LogInfo(TkDetLayers) << "theOuterWededges.size(): " << theOuterWedges.size() ;
   
  return new CompositeTECPetal(theInnerWedges,theOuterWedges);
}
