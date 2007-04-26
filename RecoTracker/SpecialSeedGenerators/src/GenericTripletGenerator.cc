#include "RecoTracker/SpecialSeedGenerators/interface/GenericTripletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace ctfseeding;


GenericTripletGenerator::GenericTripletGenerator(const edm::ParameterSet& conf): conf_(conf){
	edm::LogInfo("CtfSpecialSeedGenerator|GenericTripletGenerator") << "Constructing GenericTripletGenerator";
} 


SeedingLayerSets GenericTripletGenerator::init(const edm::EventSetup& es){
	edm::ParameterSet leyerPSet = conf_.getParameter<edm::ParameterSet>("LayerPSet");
	SeedingLayerSetsBuilder lsBuilder(leyerPSet);
  	SeedingLayerSets lss = lsBuilder.layers(es);
	return lss;	
}


const OrderedSeedingHits& GenericTripletGenerator::run(const TrackingRegion& region,
                              				     const edm::Event& e,
                              				     const edm::EventSetup& es){
	hitTriplets.clear();
	SeedingLayerSets lss = init(es);
	SeedingLayerSets::const_iterator iLss;
	for (iLss = lss.begin(); iLss != lss.end(); iLss++){
		SeedingLayers ls = *iLss;
		if (ls.size() != 3){
                	throw cms::Exception("CtfSpecialSeedGenerator") << "You are using " << ls.size() <<" layers in set instead of 3 ";
        	}	
		std::vector<SeedingHit> innerHits  = region.hits(e, es, &ls[0]);
		std::vector<SeedingHit> middleHits = region.hits(e, es, &ls[1]);
		std::vector<SeedingHit> outerHits  = region.hits(e, es, &ls[2]);
		std::vector<SeedingHit>::const_iterator iOuterHit;
		for (iOuterHit = outerHits.begin(); iOuterHit != outerHits.end(); iOuterHit++){
			std::vector<SeedingHit>::const_iterator iMiddleHit;
			for (iMiddleHit = middleHits.begin(); iMiddleHit != middleHits.end(); iMiddleHit++){
				std::vector<SeedingHit>::const_iterator iInnerHit;
				for (iInnerHit = innerHits.begin(); iInnerHit != innerHits.end(); iInnerHit++){
					hitTriplets.push_back(OrderedHitTriplet(*iInnerHit,
										*iMiddleHit,
										*iOuterHit));
				}
			}
		}
        }
	return hitTriplets;
}
