#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include <iostream>

DetLayer* NavigableLayer::detLayer() const 
{
  //cerr << "Error: NavigableLayer::detLayer() called for base class." << endl;
  return 0;
}

void      NavigableLayer::setDetLayer( DetLayer* dl) 
{
  //cerr << "Error: NavigableLayer::setDetLayer() called for base class." <<endl;
}
