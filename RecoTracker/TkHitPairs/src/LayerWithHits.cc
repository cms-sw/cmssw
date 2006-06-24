#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"

LayerWithHits::LayerWithHits( const DetLayer *dl,const SiPixelRecHitCollection::range range){
  theDetLayer = dl;
  for(SiPixelRecHitCollection::const_iterator it = range.first; it != range.second; it++){
    theHits.push_back( &(*it) );
  }
}

LayerWithHits::LayerWithHits( const DetLayer *dl,const SiStripRecHit2DLocalPosCollection::range range){
  theDetLayer = dl;
  for(SiStripRecHit2DLocalPosCollection::const_iterator it = range.first; it != range.second; it++){
    theHits.push_back( &(*it) );
  }
}

LayerWithHits::LayerWithHits( const DetLayer *dl,const SiStripRecHit2DMatchedLocalPosCollection::range range){
  theDetLayer = dl;
  for(SiStripRecHit2DMatchedLocalPosCollection::const_iterator it = range.first; it != range.second; it++){
    theHits.push_back( &(*it) );
  }
}
