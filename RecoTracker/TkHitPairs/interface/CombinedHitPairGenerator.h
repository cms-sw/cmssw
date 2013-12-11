#ifndef CombinedHitPairGenerator_H
#define CombinedHitPairGenerator_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackingRegion;
class OrderedHitPairs;
class HitPairGeneratorFromLayerPair;
namespace ctfseeding { class SeedingLayer;}
namespace edm { class Event; class EventSetup; }

/** \class CombinedHitPairGenerator
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CombinedHitPairGenerator : public HitPairGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitPairGenerator(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  CombinedHitPairGenerator(const CombinedHitPairGenerator& gen);
  virtual ~CombinedHitPairGenerator();

  /// form base class
  virtual void hitPairs( const TrackingRegion& reg, 
      OrderedHitPairs & result, const edm::Event& ev, const edm::EventSetup& es);

  /// from base class
  virtual CombinedHitPairGenerator * clone() const override;

private:
  LayerCacheType   theLayerCache;

  typedef std::vector<std::unique_ptr<HitPairGeneratorFromLayerPair> >   Container;
  Container        theGenerators;

};
#endif
