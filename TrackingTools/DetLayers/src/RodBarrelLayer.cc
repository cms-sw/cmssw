#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"

#include <algorithm>


RodBarrelLayer::RodBarrelLayer( vector<const Det*>::const_iterator firstRod,
				vector<const Det*>::const_iterator lastRod)
  : theDets(firstRod,lastRod) {}

RodBarrelLayer::RodBarrelLayer( const vector<const Det*>& theRods)
  : theDets(theRods){}


RodBarrelLayer::~RodBarrelLayer(){}

