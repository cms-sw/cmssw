#include "PixelRodBuilder.h"

using namespace edm;
using namespace std;


PixelRod* PixelRodBuilder::build(const GeometricDetPtr aRod,
				 const TrackerGeometry* theGeomDetGeometry)
{
  auto allGeometricDets = aRod->components();  

  vector<const GeomDet*> theGeomDets;
  for(auto it=allGeometricDets.begin();
	it!=allGeometricDets.end();it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );
    theGeomDets.push_back(theGeomDet);
  }
  
  return new PixelRod(theGeomDets);
}
