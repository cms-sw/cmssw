#ifndef PixelSeedLayerPairs_H
#define PixelSeedLayerPairs_H

/** \class PixelSeedLayerPairs
 * find all (resonable) pairs of pixel layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"

#include <vector>

class PixelSeedLayerPairs : public SeedLayerPairs{
public:

  PixelSeedLayerPairs():isFirstCall(true){}
  ~PixelSeedLayerPairs();

  //  virtual std::vector<LayerPair> operator()() const;
  std::vector<LayerPair> operator()() ;
  void init(const SiPixelRecHitCollection &coll,
	    const edm::EventSetup& iSetup);

private:

  //definition of the map  
  SiPixelRecHitCollection::range map_range1;
  SiPixelRecHitCollection::range map_range2;
  SiPixelRecHitCollection::range map_range3;

  SiPixelRecHitCollection::range map_diskneg1;
  SiPixelRecHitCollection::range map_diskneg2;

  SiPixelRecHitCollection::range map_diskpos1;
  SiPixelRecHitCollection::range map_diskpos2;

  TrackerLayerIdAccessor acc;
  
  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;

  LayerWithHits *pos1;
  LayerWithHits *pos2;

  LayerWithHits *neg1;
  LayerWithHits *neg2;


  std::vector<BarrelDetLayer*> bl;
  std::vector<ForwardDetLayer*> fpos;
  std::vector<ForwardDetLayer*> fneg;

  std::vector<LayerWithHits*> allLayersWithHits;
  bool isFirstCall;
};




#endif
