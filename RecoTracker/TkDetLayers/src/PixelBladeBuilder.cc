#include "PixelBladeBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/TkRotation.h"

using namespace edm;
using namespace std;
                                     
PixelBlade* PixelBladeBuilder:: build(GeometricDetPtr geometricDetFrontPanel,
				      GeometricDetPtr geometricDetBackPanel,
				      const TrackerGeometry* theGeomDetGeometry)
{
  auto frontGeometricDets = geometricDetFrontPanel->components();  
  auto backGeometricDets  = geometricDetBackPanel->components();  

  vector<const GeomDet*> theFrontGeomDets;
  vector<const GeomDet*> theBackGeomDets;

  for(auto it=frontGeometricDets.begin();
      it!=frontGeometricDets.end();it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    theFrontGeomDets.push_back(theGeomDet);
  }

  for(auto it=backGeometricDets.begin();
      it!=backGeometricDets.end();it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    theBackGeomDets.push_back(theGeomDet);
  }

  //edm::LogInfo(TkDetLayers) << "FrontGeomDet.size(): " << theFrontGeomDets.size() ;
  //edm::LogInfo(TkDetLayers) << "BackGeomDet.size():  " << theBackGeomDets.size() ;

  return new PixelBlade(theFrontGeomDets,theBackGeomDets);
}
 



