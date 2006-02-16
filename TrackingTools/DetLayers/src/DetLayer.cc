#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include <algorithm>

DetLayer::~DetLayer() { delete theNavigableLayer;}


void DetLayer::setNavigableLayer( NavigableLayer* nlp) {
  ///  delete theNavigableLayer; // bad idea!
  theNavigableLayer = nlp;

  if (nlp->detLayer() != this) {
    if (nlp->detLayer() != 0) {
      cout << "DetLayer Warning: my navigable layer does not point to me. "
	   << " Correcting..." << endl;
    }
    theNavigableLayer->setDetLayer( this);
  }
}

vector<const DetLayer*> 
DetLayer::nextLayers( const FreeTrajectoryState& fts, 
		      PropagationDirection timeDirection) const {
  return theNavigableLayer
    ? theNavigableLayer->nextLayers( fts, timeDirection)
    : vector<const DetLayer*>();
}

vector<const DetLayer*> 
DetLayer::nextLayers( PropagationDirection timeDirection) const {
  return theNavigableLayer
    ? theNavigableLayer->nextLayers( timeDirection)
    : vector<const DetLayer*>();
}


