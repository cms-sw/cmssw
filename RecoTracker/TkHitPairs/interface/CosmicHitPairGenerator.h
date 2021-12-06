#ifndef CosmicHitPairGenerator_H
#define CosmicHitPairGenerator_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
class SeedLayerPairs;
class LayerWithHits;
class DetLayer;
class TrackingRegion;

/** \class CosmicHitPairGenerator
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CosmicHitPairGenerator {
  typedef std::vector<std::unique_ptr<CosmicHitPairGeneratorFromLayerPair> > Container;

public:
  CosmicHitPairGenerator(SeedLayerPairs& layers, const TrackerGeometry&);
  CosmicHitPairGenerator(SeedLayerPairs& layers);

  ~CosmicHitPairGenerator();

  /// add generators based on layers
  //  void  add(const DetLayer* inner, const DetLayer* outer);
  void add(const LayerWithHits* inner, const LayerWithHits* outer, const TrackerGeometry& trackGeom);
  /// form base class
  void hitPairs(const TrackingRegion& reg, OrderedHitPairs& pr);

private:
  Container theGenerators;
};
#endif
