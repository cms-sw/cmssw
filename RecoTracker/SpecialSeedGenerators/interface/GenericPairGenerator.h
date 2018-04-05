#ifndef SpecialSeedGenerators_GenericPairGenerator_h
#define SpecialSeedGenerators_GenericPairGenerator_h
//FWK
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"

class SeedingLayerSetsHits;

class GenericPairGenerator : public OrderedHitsGenerator {
	public:
	GenericPairGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
	~GenericPairGenerator() override{};
	const OrderedSeedingHits& run(const TrackingRegion& region, 
					      const edm::Event & ev, 
					      const edm::EventSetup& es) override;
        void clear() override { hitPairs.clear();}
	private:
	edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;
	OrderedHitPairs hitPairs;
};


#endif
