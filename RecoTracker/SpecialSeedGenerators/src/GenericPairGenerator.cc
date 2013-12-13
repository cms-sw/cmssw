#include "RecoTracker/SpecialSeedGenerators/interface/GenericPairGenerator.h"
//#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
typedef TransientTrackingRecHit::ConstRecHitPointer SeedingHit;

#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace ctfseeding;


GenericPairGenerator::GenericPairGenerator(const edm::ParameterSet& conf): 
	theLsb(conf.getParameter<edm::ParameterSet>("LayerPSet")){
	edm::LogInfo("CtfSpecialSeedGenerator|GenericPairGenerator") << "Constructing GenericPairGenerator";
} 


SeedingLayerSets GenericPairGenerator::init(const edm::EventSetup& es){
	//edm::ParameterSet leyerPSet = conf_.getParameter<edm::ParameterSet>("LayerPSet");
	//SeedingLayerSetsBuilder lsBuilder(leyerPSet);
  	SeedingLayerSets lss = theLsb.layers(es);
	return lss;	
}


const OrderedSeedingHits& GenericPairGenerator::run(const TrackingRegion& region,
                          			    const edm::Event& e,
                              			    const edm::EventSetup& es){
	hitPairs.clear();
	hitPairs.reserve(0);
	SeedingLayerSets lss = init(es);
	SeedingLayerSets::const_iterator iLss;
	for (iLss = lss.begin(); iLss != lss.end(); iLss++){
		SeedingLayers ls = *iLss;
		if (ls.size() != 2){
                	throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << ls.size() <<" layers in set instead of 2 ";
        	}	
		std::vector<SeedingHit> innerHits  = region.hits(e, es, &ls[0]);
		std::vector<SeedingHit> outerHits  = region.hits(e, es, &ls[1]);
		std::vector<SeedingHit>::const_iterator iOuterHit;
		for (iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
			std::vector<SeedingHit>::const_iterator iInnerHit;
			for (iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
				hitPairs.push_back(OrderedHitPair(*iInnerHit,
								  *iOuterHit));
			}
		}
        }
	return hitPairs;
}
