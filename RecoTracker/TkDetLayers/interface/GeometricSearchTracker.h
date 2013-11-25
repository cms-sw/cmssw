#ifndef TkDetLayers_GeometricSearchTracker_h
#define TkDetLayers_GeometricSearchTracker_h

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

class TrackerTopology;

/** GeometricSearchTracker implementation
 *  
 */

class GeometricSearchTracker: public DetLayerGeometry {
 public:

  GeometricSearchTracker(const std::vector<BarrelDetLayer*>& pxlBar,
			 const std::vector<BarrelDetLayer*>& tib,
			 const std::vector<BarrelDetLayer*>& tob,
			 const std::vector<ForwardDetLayer*>& negPxlFwd,
			 const std::vector<ForwardDetLayer*>& negTid,
			 const std::vector<ForwardDetLayer*>& negTec,
			 const std::vector<ForwardDetLayer*>& posPxlFwd,
			 const std::vector<ForwardDetLayer*>& posTid,
			 const std::vector<ForwardDetLayer*>& posTec,
			 const TrackerTopology *tTopo);
  
  virtual ~GeometricSearchTracker();

  std::vector<DetLayer*> const & allLayers()     const {return theAllLayers;}  

  std::vector<BarrelDetLayer*>  const &  barrelLayers()  const {return theBarrelLayers;}

  std::vector<ForwardDetLayer*> const & forwardLayers() const {return theForwardLayers;}
  std::vector<ForwardDetLayer*> const & negForwardLayers() const {return theNegForwardLayers;}
  std::vector<ForwardDetLayer*> const & posForwardLayers() const {return thePosForwardLayers;}

  std::vector<BarrelDetLayer*>  const & pixelBarrelLayers() const {return thePixelBarrelLayers;}
  std::vector<BarrelDetLayer*>  const & tibLayers() const {return theTibLayers;}
  std::vector<BarrelDetLayer*>  const & tobLayers() const {return theTobLayers;}

  std::vector<ForwardDetLayer*> const & negPixelForwardLayers() const {return theNegPixelForwardLayers;}
  std::vector<ForwardDetLayer*> const &  negTidLayers() const {return theNegTidLayers;}
  std::vector<ForwardDetLayer*> const &  negTecLayers() const {return theNegTecLayers;}

  std::vector<ForwardDetLayer*> const &  posPixelForwardLayers() const {return thePosPixelForwardLayers;}
  std::vector<ForwardDetLayer*> const &  posTidLayers() const {return thePosTidLayers;}
  std::vector<ForwardDetLayer*> const &  posTecLayers() const {return thePosTecLayers;}

  
  /// Give the DetId of a module, returns the pointer to the corresponding DetLayer
  virtual const DetLayer* idToLayer(const DetId& detId) const;

  /// obsolete method. Use idToLayer() instead.
  const DetLayer*   detLayer( const DetId& id) const {return idToLayer(id);};

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

  const TrackerTopology *theTrkTopo;
};


#endif 
