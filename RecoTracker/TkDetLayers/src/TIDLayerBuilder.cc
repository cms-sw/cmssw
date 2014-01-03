#include "TIDLayerBuilder.h"
#include "TIDRingBuilder.h"

using namespace edm;
using namespace std;

TIDLayer* TIDLayerBuilder::build(const GeometricDetPtr aTIDLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  auto theGeometricRings = aTIDLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricRings.size(): " << theGeometricRings.size() ;

  TIDRingBuilder myBuilder;
  vector<const TIDRing*> theTIDRings;

  for(auto it=theGeometricRings.cbegin();
      it!=theGeometricRings.cend();it++){
    theTIDRings.push_back(myBuilder.build( *it,theGeomDetGeometry));    
  }

  return new TIDLayer(theTIDRings);
}
