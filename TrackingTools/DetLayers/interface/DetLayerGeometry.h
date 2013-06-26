#ifndef DetLayers_DetLayerGeometry_h
#define DetLayers_DetLayerGeometry_h

/**
 * Base class for "reconstruction" geometries. Given the detId of a module, 
 * the class has methods to access the corresponding module's DetLayer.
 * This base class returns always a dummy zero pointer to the DetLayers. Its 
 * methods are virtual and are supposed to be overridden by Muon, Tracker 
 * and Global concrete reconstruction geometries.
 *
 * \author Boris Mangano (UCSD)  2/6/2009
 */
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

class DetLayer;

class DetLayerGeometry {
 public:
  DetLayerGeometry(){};
	
	virtual ~DetLayerGeometry() {} 
 
  /*
  const std::vector<DetLayer*>& allLayers() const =0;
  const std::vector<DetLayer*>& barrelLayers() const =0;
  const std::vector<DetLayer*>& negForwardLayers() const =0;
  const std::vector<DetLayer*>& posForwardLayers() const =0;
  */


  /// Give the DetId of a module, returns the pointer to the corresponding DetLayer
  /// This method is dummy and has to be overridden in another derived class
  virtual const DetLayer* idToLayer(const DetId& detId) const {return 0;}
 
};


#endif
