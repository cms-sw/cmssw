#include "RecoTracker/TkDetLayers/interface/PixelForwardLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/PixelBladeBuilder.h"

PixelForwardLayer* PixelForwardLayerBuilder::build(const GeometricDet* aPixelForwardLayer,
						   const TrackingGeometry* theGeomDetGeometry){
  vector<const GeometricDet*>  theGeometricPanels = aPixelForwardLayer->components();
  int panelsSize = theGeometricPanels.size();

  /*
  for(vector<const GeometricDet*>::const_iterator it= theGeometricPanels.begin(); 
      it!=theGeometricPanels.end();it++){
    
    cout << "panel.phi(): " << (*it)->positionBounds().phi() << " , " 
	 << "panel.z():   " << (*it)->positionBounds().z()   << " , "
	 << "comp.size(): " << (*it)->components().size()    << endl;    
  }
  */

  //cout << "pixelFwdLayer.panels().size(): " << panelsSize << endl;  

  vector<const PixelBlade*> theBlades;
  PixelBladeBuilder myBladeBuilder;

  for(int i=0; i< (panelsSize/2); i++){
    theBlades.push_back( myBladeBuilder.build( theGeometricPanels[i],
					       theGeometricPanels[i+(panelsSize/2)],
					       theGeomDetGeometry ) );
  }
  
  return new PixelForwardLayer(theBlades);  
}
