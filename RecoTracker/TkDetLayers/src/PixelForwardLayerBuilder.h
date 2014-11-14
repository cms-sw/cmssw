#ifndef TkDetLayers_PixelForwardLayerBuilder_h
#define TkDetLayers_PixelForwardLayerBuilder_h


#include "PixelForwardLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "PixelForwardLayerPhase1.h"
#include "PixelForwardLayer.h"
#include "PixelBladeBuilder.h"

using namespace edm;
using namespace std;

/** A concrete builder for PixelForwardLayer 
 */

#pragma GCC visibility push(hidden)
template <class T1, class T2>
class PixelForwardLayerBuilder {  
 public:
  PixelForwardLayerBuilder(){};
  ForwardDetLayer* build(const GeometricDet* aPixelForwardLayer,
			 const TrackerGeometry* theGeomDetGeometry) __attribute__ ((cold));

  
};

template <class T1, class T2>
  ForwardDetLayer* PixelForwardLayerBuilder<T1,T2>::build(const GeometricDet* aPixelForwardLayer,
							  const TrackerGeometry* theGeomDetGeometry) {
  vector<const GeometricDet*>  theGeometricPanels = aPixelForwardLayer->components();
  int panelsSize = theGeometricPanels.size();

  /*
  int num = 0;
  for(vector<const GeometricDet*>::const_iterator it= theGeometricPanels.begin();
      it!=theGeometricPanels.end(); it++, ++num) {
    edm::LogInfo("TkDetLayers") << "PanelsSize: " << panelsSize << " , "
                                << "PanelNum: " << num  << " , "
                                << "panel.phi(): " << (*it)->positionBounds().phi()  << " , "
                                << "panel.z():   " << (*it)->positionBounds().z()    << " , "
                                << "panel.y():   " << (*it)->positionBounds().y() << " , "
                                << "panel.x():   " << (*it)->positionBounds().x() << " , "
                                << "panel.r():   " << (*it)->positionBounds().perp() << " , "
                                << "panel.rmax():   " << (*it)->bounds()->rSpan().second << " , "
                                << "comp.size(): " << (*it)->components().size();
  }
  */

  //edm::LogInfo(TkDetLayers) << "pixelFwdLayer.panels().size(): " << panelsSize ;

  vector<const T1*> theBlades;
  PixelBladeBuilder<T1> myBladeBuilder;

  for(int i=0; i< (panelsSize/2); i++) {
    theBlades.push_back( myBladeBuilder.build( theGeometricPanels[i],
					       theGeometricPanels[i+(panelsSize/2)],
					       theGeomDetGeometry ) );
  }

  return new T2(theBlades);
}

#pragma GCC visibility pop
#endif 
