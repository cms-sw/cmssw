#ifndef TkDetLayers_GeometricSearchTracker_h
#define TkDetLayers_GeometricSearchTracker_h

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"


/** GeometricSearchTracker implementation
 *  
 */

class GeometricSearchTracker {
 public:

  GeometricSearchTracker(const std::vector<BarrelDetLayer*>& pxlBar,
			 const std::vector<BarrelDetLayer*>& tib,
			 const std::vector<BarrelDetLayer*>& tob,
			 const std::vector<ForwardDetLayer*>& negPxlFwd,
			 const std::vector<ForwardDetLayer*>& negTid,
			 const std::vector<ForwardDetLayer*>& negTec,
			 const std::vector<ForwardDetLayer*>& posPxlFwd,
			 const std::vector<ForwardDetLayer*>& posTid,
			 const std::vector<ForwardDetLayer*>& posTec);
  
  ~GeometricSearchTracker();

  std::vector<DetLayer*>        allLayers()     const {return theAllLayers;}  

  std::vector<BarrelDetLayer*>  barrelLayers()  const {return theBarrelLayers;}

  std::vector<ForwardDetLayer*> forwardLayers() const {return theForwardLayers;}
  std::vector<ForwardDetLayer*> negForwardLayers() const {return theNegForwardLayers;}
  std::vector<ForwardDetLayer*> posForwardLayers() const {return thePosForwardLayers;}

  std::vector<BarrelDetLayer*>  pixelBarrelLayers() const {return thePixelBarrelLayers;}
  std::vector<BarrelDetLayer*>  tibLayers() const {return theTibLayers;}
  std::vector<BarrelDetLayer*>  tobLayers() const {return theTobLayers;}

  std::vector<ForwardDetLayer*>  negPixelForwardLayers() const {return theNegPixelForwardLayers;}
  std::vector<ForwardDetLayer*>  negTidLayers() const {return theNegTidLayers;}
  std::vector<ForwardDetLayer*>  negTecLayers() const {return theNegTecLayers;}

  std::vector<ForwardDetLayer*>  posPixelForwardLayers() const {return thePosPixelForwardLayers;}
  std::vector<ForwardDetLayer*>  posTidLayers() const {return thePosTidLayers;}
  std::vector<ForwardDetLayer*>  posTecLayers() const {return thePosTecLayers;}

  const DetLayer*          detLayer( const DetId& id) const;

 private:
  std::vector<DetLayer*>        theAllLayers;
  std::vector<BarrelDetLayer*>  theBarrelLayers;
  std::vector<ForwardDetLayer*> theForwardLayers;
  std::vector<ForwardDetLayer*> theNegForwardLayers;
  std::vector<ForwardDetLayer*> thePosForwardLayers;

  std::vector<BarrelDetLayer*>  thePixelBarrelLayers;
  std::vector<BarrelDetLayer*>  theTibLayers;
  std::vector<BarrelDetLayer*>  theTobLayers;

  std::vector<ForwardDetLayer*> theNegPixelForwardLayers;
  std::vector<ForwardDetLayer*> theNegTidLayers;
  std::vector<ForwardDetLayer*> theNegTecLayers;
  std::vector<ForwardDetLayer*> thePosPixelForwardLayers;
  std::vector<ForwardDetLayer*> thePosTidLayers;
  std::vector<ForwardDetLayer*> thePosTecLayers;
};


#endif 
