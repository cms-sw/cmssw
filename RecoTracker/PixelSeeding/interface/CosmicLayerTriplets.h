#ifndef RecoTracker_PixelSeeding_CosmicLayerTriplets_h
#define RecoTracker_PixelSeeding_CosmicLayerTriplets_h

/** \class CosmicLayerTriplets
 * find all (resonable) pairs of pixel layers
 */
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
//#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

class GeometricSearchTracker;
class TrackerTopology;

#include <vector>
class CosmicLayerTriplets {
public:
  CosmicLayerTriplets(std::string geometry,
                      const SiStripRecHit2DCollection &collrphi,
                      const GeometricSearchTracker &track,
                      const TrackerTopology &ttopo) {
    init(collrphi, std::move(geometry), track, ttopo);
  };
  ~CosmicLayerTriplets();
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);
  typedef std::pair<SeedLayerPairs::LayerPair, std::vector<const LayerWithHits *> > LayerPairAndLayers;

  //  virtual std::vector<LayerPair> operator()() const;
  //  std::vector<LayerTriplet> operator()() ;
  std::vector<LayerPairAndLayers> layers();

private:
  //definition of the map

  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;
  LayerWithHits *lh4;

  std::vector<BarrelDetLayer const *> bl;
  //MP
  std::vector<LayerWithHits *> allLayersWithHits;

  void init(const SiStripRecHit2DCollection &collrphi,
            std::string geometry,
            const GeometricSearchTracker &track,
            const TrackerTopology &ttopo);

private:
  std::string _geometry;
};

#endif
