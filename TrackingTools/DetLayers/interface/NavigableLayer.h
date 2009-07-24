#ifndef DetLayers_NavigableLayer_h
#define DetLayers_NavigableLayer_h

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"

#include <vector>

class DetLayer;
class FreeTrajectoryState;

/** The navigation component of the DetLayer.
 *  If navigation is not setup the DetLayer has a zero pointer to
 *  a NavigableLayer; when navigation is setup each DetLayer is
 *  given an instance of NavigableLayer that implements the 
 *  navigation algorithm. Navigation requests to the DetLayer are
 *  forwarded to the navigable layer.
 */

class NavigableLayer  {
public:

  virtual ~NavigableLayer() {}

  virtual std::vector<const DetLayer*> 
  nextLayers( NavigationDirection direction) const = 0;

  virtual std::vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const = 0;

  virtual std::vector<const DetLayer*> 
  compatibleLayers( NavigationDirection direction) const = 0;

  virtual std::vector<const DetLayer*> 
  compatibleLayers( const FreeTrajectoryState& fts, 
		    PropagationDirection timeDirection) const = 0;

  virtual DetLayer* detLayer() const = 0;
  virtual void   setDetLayer( DetLayer* dl) = 0;

};

#endif 
