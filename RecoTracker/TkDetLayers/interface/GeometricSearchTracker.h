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

  GeometricSearchTracker(const std::vector<BarrelDetLayer const*>& pxlBar,
			 const std::vector<BarrelDetLayer const*>& tib,
			 const std::vector<BarrelDetLayer const*>& tob,
			 const std::vector<ForwardDetLayer const*>& negPxlFwd,
			 const std::vector<ForwardDetLayer const*>& negTid,
			 const std::vector<ForwardDetLayer const*>& negTec,
			 const std::vector<ForwardDetLayer const*>& posPxlFwd,
			 const std::vector<ForwardDetLayer const*>& posTid,
			 const std::vector<ForwardDetLayer const*>& posTec,
			 const TrackerTopology* tTopo) __attribute__ ((cold));
  
  virtual ~GeometricSearchTracker() __attribute__ ((cold));

  std::vector<DetLayer const*> const & allLayers()     const {return theAllLayers;}  

  std::vector<BarrelDetLayer const*>  const &  barrelLayers()  const {return theBarrelLayers;}

  std::vector<ForwardDetLayer const*> const & forwardLayers() const {return theForwardLayers;}
  std::vector<ForwardDetLayer const*> const & negForwardLayers() const {return theNegForwardLayers;}
  std::vector<ForwardDetLayer const*> const & posForwardLayers() const {return thePosForwardLayers;}

  std::vector<BarrelDetLayer const*>  const & pixelBarrelLayers() const {return thePixelBarrelLayers;}
  std::vector<BarrelDetLayer const*>  const & tibLayers() const {return theTibLayers;}
  std::vector<BarrelDetLayer const*>  const & tobLayers() const {return theTobLayers;}

  std::vector<ForwardDetLayer const*> const & negPixelForwardLayers() const {return theNegPixelForwardLayers;}
  std::vector<ForwardDetLayer const*> const &  negTidLayers() const {return theNegTidLayers;}
  std::vector<ForwardDetLayer const*> const &  negTecLayers() const {return theNegTecLayers;}

  std::vector<ForwardDetLayer const*> const &  posPixelForwardLayers() const {return thePosPixelForwardLayers;}
  std::vector<ForwardDetLayer const*> const &  posTidLayers() const {return thePosTidLayers;}
  std::vector<ForwardDetLayer const*> const &  posTecLayers() const {return thePosTecLayers;}

  
  /// Give the DetId of a module, returns the pointer to the corresponding DetLayer
  virtual const DetLayer* idToLayer(const DetId& detId) const;

  /// obsolete method. Use idToLayer() instead.
  const DetLayer*   detLayer( const DetId& id) const {return idToLayer(id);};

 private:
  std::vector<DetLayer const*>        theAllLayers;
  std::vector<BarrelDetLayer const*>  theBarrelLayers;
  std::vector<ForwardDetLayer const*> theForwardLayers;
  std::vector<ForwardDetLayer const*> theNegForwardLayers;
  std::vector<ForwardDetLayer const*> thePosForwardLayers;

  std::vector<BarrelDetLayer const*>  thePixelBarrelLayers;
  std::vector<BarrelDetLayer const*>  theTibLayers;
  std::vector<BarrelDetLayer const*>  theTobLayers;

  std::vector<ForwardDetLayer const*> theNegPixelForwardLayers;
  std::vector<ForwardDetLayer const*> theNegTidLayers;
  std::vector<ForwardDetLayer const*> theNegTecLayers;
  std::vector<ForwardDetLayer const*> thePosPixelForwardLayers;
  std::vector<ForwardDetLayer const*> thePosTidLayers;
  std::vector<ForwardDetLayer const*> thePosTecLayers;

  const TrackerTopology *theTrkTopo;
};


#endif 
