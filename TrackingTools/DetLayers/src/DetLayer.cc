#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;

DetLayer::~DetLayer() { delete theNavigableLayer;}


void DetLayer::setNavigableLayer( NavigableLayer* nlp) {
  ///  delete theNavigableLayer; // bad idea!
  theNavigableLayer = nlp;

  //nlp=0 amount to cleaning the link. do nothing further
  if (nlp){
  if (nlp->detLayer() != this) {
    if (nlp->detLayer() != 0) {
      edm::LogWarning("DetLayers") << "DetLayer Warning: my navigable layer does not point to me. "
				   << " Correcting..." ;
    }
    theNavigableLayer->setDetLayer( this);
  }
  }//nlp!=0
}

