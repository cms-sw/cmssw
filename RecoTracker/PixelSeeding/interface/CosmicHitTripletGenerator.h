#ifndef RecoTracker_PixelSeeding_CosmicHitTripletGenerator_h
#define RecoTracker_PixelSeeding_CosmicHitTripletGenerator_h

#include <vector>
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplets.h"
#include "RecoTracker/PixelSeeding/interface/CosmicHitTripletGeneratorFromLayerTriplet.h"
#include "DataFormats/Common/interface/RangeMap.h"

class LayerWithHits;
class DetLayer;
class TrackingRegion;
class CosmicLayerTriplets;

/** \class CosmicHitTripletGenerator
 * Hides set of HitTripletGeneratorFromLayerTriplet generators.
 */

class CosmicHitTripletGenerator {
  typedef std::vector<std::unique_ptr<CosmicHitTripletGeneratorFromLayerTriplet> > Container;

public:
  CosmicHitTripletGenerator(CosmicLayerTriplets& layers, const TrackerGeometry& trackGeom);
  CosmicHitTripletGenerator(CosmicLayerTriplets& layers);

  ~CosmicHitTripletGenerator();

  /// add generators based on layers
  //  void  add(const DetLayer* inner, const DetLayer* outer);
  void add(const LayerWithHits* inner,
           const LayerWithHits* middle,
           const LayerWithHits* outer,
           const TrackerGeometry& trackGeom);

  void hitTriplets(const TrackingRegion& reg, OrderedHitTriplets& prs);

private:
  Container theGenerators;
};
#endif
