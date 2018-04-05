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

  typedef std::vector<std::unique_ptr<CosmicHitPairGeneratorFromLayerPair> >   Container;

public:
  CosmicHitPairGenerator(SeedLayerPairs& layers, const edm::EventSetup& iSetup);
  CosmicHitPairGenerator(SeedLayerPairs& layers);


  ~CosmicHitPairGenerator();

  /// add generators based on layers
    //  void  add(const DetLayer* inner, const DetLayer* outer);
    void  add(const LayerWithHits* inner, 
	      const LayerWithHits* outer,
	      const edm::EventSetup& iSetup);
  /// form base class
  void hitPairs( const TrackingRegion& reg,
		 OrderedHitPairs & prs,
		 const edm::EventSetup& iSetup);
private:


  Container        theGenerators;

};
#endif
