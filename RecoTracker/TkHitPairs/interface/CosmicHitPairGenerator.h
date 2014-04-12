#ifndef CosmicHitPairGenerator_H
#define CosmicHitPairGenerator_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
class SeedLayerPairs;
class LayerWithHits;
class DetLayer;
class TrackingRegion;
class HitPairGeneratorFromLayerPair;


/** \class CosmicHitPairGenerator
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CosmicHitPairGenerator : public HitPairGenerator{

  typedef std::vector<CosmicHitPairGeneratorFromLayerPair *>   Container;

public:
  CosmicHitPairGenerator(SeedLayerPairs& layers, const edm::EventSetup& iSetup);
  CosmicHitPairGenerator(SeedLayerPairs& layers);


  ~CosmicHitPairGenerator();

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) override {}

  /// add generators based on layers
    //  void  add(const DetLayer* inner, const DetLayer* outer);
    void  add(const LayerWithHits* inner, 
	      const LayerWithHits* outer,
	      const edm::EventSetup& iSetup);
  /// form base class
  virtual void hitPairs( const TrackingRegion& reg, 
			 OrderedHitPairs & prs, 
			 const edm::EventSetup& iSetup);
  virtual void hitPairs( const TrackingRegion& reg, 
			 OrderedHitPairs & prs, 
                   const edm::Event & ev,
			 const edm::EventSetup& iSetup) {}

  /// from base class
  virtual CosmicHitPairGenerator * clone() const 
    { return new CosmicHitPairGenerator(*this); }

private:


  Container        theGenerators;

};
#endif
