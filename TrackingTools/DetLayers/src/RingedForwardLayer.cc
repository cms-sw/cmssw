#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetRing.h"
#include <algorithm>

RingedForwardLayer::RingedForwardLayer( vector<const Det*>::const_iterator firstRing,
					vector<const Det*>::const_iterator lastRing)
  : theDets(firstRing,lastRing) {}

RingedForwardLayer::RingedForwardLayer( const vector<const Det*>& theRings)
  : theDets(theRings){}


RingedForwardLayer::~RingedForwardLayer(){}


/*    
Module RingedForwardLayer::module() const 
{ 
  return theDets.front()->detUnits().front()->type().module();
}
*/
