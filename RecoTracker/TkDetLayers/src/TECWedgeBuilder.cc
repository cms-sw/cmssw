#include "RecoTracker/TkDetLayers/interface/TECWedgeBuilder.h"
#include "RecoTracker/TkDetLayers/interface/CompositeTECWedge.h"
#include "RecoTracker/TkDetLayers/interface/SimpleTECWedge.h"

TECWedge* TECWedgeBuilder::build(const GeometricDet* aTECWedge,
				 const TrackingGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricDets = aTECWedge->components();
  //cout << "theGeometricDets.size(): " << theGeometricDets.size() << endl;

  if(theGeometricDets.size() == 1 ) {
    const GeomDet* theGeomDet = 
      theGeomDetGeometry->idToDet( theGeometricDets.front()->geographicalID() );
    return new SimpleTECWedge(theGeomDet);
  }

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;

  //---- to evaluate meanZ
  double meanZ = 0;
  for(vector<const GeometricDet*>::const_iterator it=theGeometricDets.begin();
      it!=theGeometricDets.end();it++){
    meanZ = meanZ + (*it)->positionBounds().z();
  }

  meanZ = meanZ/theGeometricDets.size();
  //cout << "meanZ: " << meanZ << endl;
  //----

  for(vector<const GeometricDet*>::const_iterator it=theGeometricDets.begin();
      it!=theGeometricDets.end();it++){
    //cout << "GeometricDet.z(): " << (*it)->positionBounds().z() << endl;
    //cout << "GeometricDet.perp(): " << (*it)->positionBounds().perp() << endl;
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    //cout << "GeomDet.z(): " << theGeomDet->surface().position().z() << endl;
    //cout << "GeomDet.perp(): " << theGeomDet->surface().position().perp() << endl;


    if( fabs( (*it)->positionBounds().z() ) < fabs(meanZ))
      innerGeomDets.push_back(theGeomDet);

    if( fabs( (*it)->positionBounds().z() ) > fabs(meanZ))
      outerGeomDets.push_back(theGeomDet);      
  }

  //cout << "innerGeomDets.size(): " << innerGeomDets.size() << endl;
  //cout << "outerGeomDets.size(): " << outerGeomDets.size() << endl;

  return new CompositeTECWedge(innerGeomDets,outerGeomDets);
}
