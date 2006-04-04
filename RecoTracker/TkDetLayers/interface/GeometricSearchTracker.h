#ifndef TkDetLayers_GeometricSearchTracker_h
#define TkDetLayers_GeometricSearchTracker_h

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"


/** GeometricSearchTracker implementation
 *  
 */

class GeometricSearchTracker {
 public:

  GeometricSearchTracker(const vector<BarrelDetLayer*>& pxlBar,
			 const vector<BarrelDetLayer*>& tib,
			 const vector<BarrelDetLayer*>& tob,
			 const vector<ForwardDetLayer*>& negPxlFwd,
			 const vector<ForwardDetLayer*>& negTid,
			 const vector<ForwardDetLayer*>& negTec,
			 const vector<ForwardDetLayer*>& posPxlFwd,
			 const vector<ForwardDetLayer*>& posTid,
			 const vector<ForwardDetLayer*>& posTec);
  
  ~GeometricSearchTracker();

  vector<DetLayer*>        allLayers()     const {return theAllLayers;}  

  vector<BarrelDetLayer*>  barrelLayers()  const {return theBarrelLayers;}

  vector<ForwardDetLayer*> forwardLayers() const {return theForwardLayers;}
  vector<ForwardDetLayer*> negForwardLayers() const {return theNegForwardLayers;}
  vector<ForwardDetLayer*> posForwardLayers() const {return thePosForwardLayers;}

  vector<BarrelDetLayer*>  pixelBarrelLayers() const {return thePixelBarrelLayers;}
  vector<BarrelDetLayer*>  tibLayers() const {return theTibLayers;}
  vector<BarrelDetLayer*>  tobLayers() const {return theTobLayers;}

  vector<ForwardDetLayer*>  negPixelForwardLayers() const {return theNegPixelForwardLayers;}
  vector<ForwardDetLayer*>  negTidLayers() const {return theNegTidLayers;}
  vector<ForwardDetLayer*>  negTecLayers() const {return theNegTecLayers;}

  vector<ForwardDetLayer*>  posPixelForwardLayers() const {return thePosPixelForwardLayers;}
  vector<ForwardDetLayer*>  posTidLayers() const {return thePosTidLayers;}
  vector<ForwardDetLayer*>  posTecLayers() const {return thePosTecLayers;}

  const DetLayer*          detLayer( const DetId& id) const;

 private:
  vector<DetLayer*>        theAllLayers;
  vector<BarrelDetLayer*>  theBarrelLayers;
  vector<ForwardDetLayer*> theForwardLayers;
  vector<ForwardDetLayer*> theNegForwardLayers;
  vector<ForwardDetLayer*> thePosForwardLayers;

  vector<BarrelDetLayer*>  thePixelBarrelLayers;
  vector<BarrelDetLayer*>  theTibLayers;
  vector<BarrelDetLayer*>  theTobLayers;

  vector<ForwardDetLayer*> theNegPixelForwardLayers;
  vector<ForwardDetLayer*> theNegTidLayers;
  vector<ForwardDetLayer*> theNegTecLayers;
  vector<ForwardDetLayer*> thePosPixelForwardLayers;
  vector<ForwardDetLayer*> thePosTidLayers;
  vector<ForwardDetLayer*> thePosTecLayers;
};


#endif 
