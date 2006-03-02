#include "RecoTracker/TkDetLayers/interface/TECPetalBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TECWedgeBuilder.h"
#include "RecoTracker/TkDetLayers/interface/CompositeTECPetal.h"

TECPetal* TECPetalBuilder::build(const GeometricDet* aTECPetal,
				 const TrackingGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricWedges = aTECPetal->components();
  //cout << "theGeometricWedges.size(): " << theGeometricWedges.size() << endl;
  
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
  
  //cout << "theInnerWededges.size(): " << theInnerWedges.size() << endl;
  //cout << "theOuterWededges.size(): " << theOuterWedges.size() << endl;
   
  return new CompositeTECPetal(theInnerWedges,theOuterWedges);
}
