#ifndef LayerWithHits_H
#define LayerWithHits_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"
class LayerWithHits
{
 public:

   LayerWithHits( const PixelBarrelLayer *dl, edm::RangeMap<DetId, edm::OwnVector<SiPixelRecHitCollection,edm::ClonePolicy<SiPixelRecHitCollection> >, edm::ClonePolicy<SiPixelRecHitCollection> >::range ran):
  ddl(dl),RANGE(ran)
  {std::cout<<"za";};



  ~LayerWithHits(){std::cout<<"ze";};

  const PixelBarrelLayer* layer() {return ddl;};


 private:
  const PixelBarrelLayer* ddl;
  edm::RangeMap<DetId, edm::OwnVector<SiPixelRecHitCollection,edm::ClonePolicy<SiPixelRecHitCollection> >, edm::ClonePolicy<SiPixelRecHitCollection> >::range RANGE;

};
#endif

