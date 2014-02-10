#ifndef CombinedHitPairGenerator_H
#define CombinedHitPairGenerator_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"


class TrackingRegion;
class OrderedHitPairs;
class HitPairGeneratorFromLayerPair;
class SeedingLayerSetsHits;
namespace edm { class Event; class EventSetup; }

#include <memory>

/** \class CombinedHitPairGenerator
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CombinedHitPairGenerator : public HitPairGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitPairGenerator(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  virtual ~CombinedHitPairGenerator();

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) override;

  /// form base class
  virtual void hitPairs( const TrackingRegion& reg, 
      OrderedHitPairs & result, const edm::Event& ev, const edm::EventSetup& es);

  /// from base class
  virtual CombinedHitPairGenerator * clone() const 
    { return new CombinedHitPairGenerator(*this); }

private:
  CombinedHitPairGenerator(const CombinedHitPairGenerator & cb); 

  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;

  LayerCacheType   theLayerCache;

  std::unique_ptr<HitPairGeneratorFromLayerPair> theGenerator;

};
#endif
