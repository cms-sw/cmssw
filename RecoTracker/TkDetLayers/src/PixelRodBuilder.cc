#include "RecoTracker/TkDetLayers/interface/PixelRodBuilder.h"

PixelRod* PixelRodBuilder::build(const GeometricDet* aRod,
				 const TrackingGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*> allGeometricDets = aRod->components();  

  vector<const GeomDet*> theGeomDets;
  for(vector<const GeometricDet*>::iterator it=allGeometricDets.begin();
	it!=allGeometricDets.end();it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    theGeomDets.push_back(theGeomDet);
  }
  
  return new PixelRod(theGeomDets);
}
