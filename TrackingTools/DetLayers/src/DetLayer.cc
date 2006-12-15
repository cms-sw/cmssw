#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;

DetLayer::~DetLayer() { delete theNavigableLayer;}


void DetLayer::setNavigableLayer( NavigableLayer* nlp) {
  ///  delete theNavigableLayer; // bad idea!
  theNavigableLayer = nlp;

  if (nlp->detLayer() != this) {
    if (nlp->detLayer() != 0) {
      edm::LogWarning("DetLayers") << "DetLayer Warning: my navigable layer does not point to me. "
				   << " Correcting..." ;
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
DetLayer::nextLayers( NavigationDirection direction) const {
  return theNavigableLayer
    ? theNavigableLayer->nextLayers( direction)
    : vector<const DetLayer*>();
}

vector<const DetLayer*> 
DetLayer::compatibleLayers( const FreeTrajectoryState& fts, 
			    PropagationDirection timeDirection) const {
  return theNavigableLayer
    ? theNavigableLayer->compatibleLayers( fts, timeDirection)
    : vector<const DetLayer*>();
}

vector<const DetLayer*> 
DetLayer::compatibleLayers( PropagationDirection timeDirection) const {
  return theNavigableLayer
    ? theNavigableLayer->compatibleLayers( timeDirection)
    : vector<const DetLayer*>();
}


