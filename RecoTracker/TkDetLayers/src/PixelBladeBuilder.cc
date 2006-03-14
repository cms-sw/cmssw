#include "RecoTracker/TkDetLayers/interface/PixelBladeBuilder.h"
#include "Geometry/Surface/interface/TkRotation.h"
                                     
PixelBlade* PixelBladeBuilder:: build(const GeometricDet* geometricDetFrontPanel,
				      const GeometricDet* geometricDetBackPanel,
				      const TrackingGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*> frontGeometricDets = geometricDetFrontPanel->components();  
  vector<const GeometricDet*> backGeometricDets  = geometricDetBackPanel->components();  

  vector<const GeomDet*> theFrontGeomDets;
  vector<const GeomDet*> theBackGeomDets;

  for(vector<const GeometricDet*>::iterator it=frontGeometricDets.begin();
      it!=frontGeometricDets.end();it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    theFrontGeomDets.push_back(theGeomDet);
  }

  for(vector<const GeometricDet*>::iterator it=backGeometricDets.begin();
      it!=backGeometricDets.end();it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    theBackGeomDets.push_back(theGeomDet);
  }

  //cout << "FrontGeomDet.size(): " << theFrontGeomDets.size() << endl;
  //cout << "BackGeomDet.size():  " << theBackGeomDets.size() << endl;

  return new PixelBlade(theFrontGeomDets,theBackGeomDets);
}
 



