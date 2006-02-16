#ifndef DetLayers_NavigableLayer_h
#define DetLayers_NavigableLayer_h

#include "TrackingTools/DetLayers/interface/Enumerators.h"
#include <vector>

class DetLayer;
class FreeTrajectoryState;

using namespace std;

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

  virtual vector<const DetLayer*> 
  nextLayers( PropagationDirection timeDirection) const = 0;

  virtual vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const = 0;

  virtual DetLayer* detLayer() const;
  virtual void   setDetLayer( DetLayer* dl);

};

#endif 
