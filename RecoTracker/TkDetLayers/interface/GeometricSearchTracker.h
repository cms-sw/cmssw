#ifndef TkDetLayers_GeometricSearchTracker_h
#define TkDetLayers_GeometricSearchTracker_h

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"


/** GeometricSearchTracker implementation
 *  
 */

class GeometricSearchTracker {
 public:

  GeometricSearchTracker(const vector<BarrelDetLayer*>& bl,
			 const vector<ForwardDetLayer*>& negFl,
			 const vector<ForwardDetLayer*>& posFl);
  
  ~GeometricSearchTracker();
  
  vector<BarrelDetLayer*>  barrelLayers()  const {return theBarrelLayers;}
  vector<ForwardDetLayer*> negForwardLayers() const {return theNegForwardLayers;}
  vector<ForwardDetLayer*> posForwardLayers() const {return thePosForwardLayers;}
  vector<ForwardDetLayer*> forwardLayers() const {return theForwardLayers;}
  vector<DetLayer*>        allLayers()     const {return theAllLayers;}

 private:
  vector<BarrelDetLayer*>  theBarrelLayers;
  vector<ForwardDetLayer*> theNegForwardLayers;
  vector<ForwardDetLayer*> thePosForwardLayers;
  vector<ForwardDetLayer*> theForwardLayers;
  vector<DetLayer*>        theAllLayers;
};


#endif 
