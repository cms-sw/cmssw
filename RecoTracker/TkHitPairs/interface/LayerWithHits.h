#ifndef LayerWithHits_H
#define LayerWithHits_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"
class LayerWithHits
{
 public:



  LayerWithHits( const PixelBarrelLayer *dl,const SiPixelRecHitCollection::range ran): ddl(dl),RANGE(ran){};



  ~LayerWithHits(){};

const  PixelBarrelLayer* layer()  const {return ddl;};
SiPixelRecHitCollection::range Range()const {return RANGE;};

 private:
 const  PixelBarrelLayer* ddl;

  const    SiPixelRecHitCollection::range RANGE;
};
#endif

