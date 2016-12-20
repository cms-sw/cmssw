#include "RecoTracker/SpecialSeedGenerators/interface/GenericPairGenerator.h"
//#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
typedef SeedingHitSet::ConstRecHitPointer SeedingHit;

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
using namespace ctfseeding;


GenericPairGenerator::GenericPairGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector& iC):
  theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(conf.getParameter<edm::InputTag>("LayerSrc"))) {
	edm::LogInfo("CtfSpecialSeedGenerator|GenericPairGenerator") << "Constructing GenericPairGenerator";
} 


const OrderedSeedingHits& GenericPairGenerator::run(const TrackingRegion& region,
                          			    const edm::Event& e,
                              			    const edm::EventSetup& es){
	hitPairs.clear();
	hitPairs.reserve(0);
        edm::Handle<SeedingLayerSetsHits> hlayers;
        e.getByToken(theSeedingLayerToken, hlayers);
        const SeedingLayerSetsHits& layers = *hlayers;
        if(layers.numberOfLayersInSet() != 2)
          throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << layers.numberOfLayersInSet() <<" layers in set instead of 2 ";

        for(SeedingLayerSetsHits::SeedingLayerSet ls: layers) {
		auto innerHits  = region.hits(e, es, ls[0]);
		auto outerHits  = region.hits(e, es, ls[1]);
		for (auto iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
		  for (auto iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
		    hitPairs.push_back(OrderedHitPair(&(**iInnerHit),
						      &(**iOuterHit))
				       );
		  }
		}
        }
	return hitPairs;
}
