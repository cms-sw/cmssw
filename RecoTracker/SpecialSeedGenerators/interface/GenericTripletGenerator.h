#ifndef SpecialSeedGenerators_GenericTripletGenerator_h
#define SpecialSeedGenerators_GenericTripletGenerator_h
//FWK
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"

//class ctfseeding::SeedingLayers;

class GenericTripletGenerator : public OrderedHitsGenerator {
	public:
	GenericTripletGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
	virtual ~GenericTripletGenerator(){};
	virtual const OrderedSeedingHits& run(const TrackingRegion& region, 
					      const edm::Event & ev, 
					      const edm::EventSetup& es);
        void clear() {hitTriplets.clear();}
	private:
	std::pair<bool,float> qualityFilter(const OrderedHitTriplet& oht, 
					    const std::map<float, OrderedHitTriplet>& map,
					    const ctfseeding::SeedingLayers& ls) const;
	SeedingLayerSetsBuilder theLsb;	
	ctfseeding::SeedingLayerSets theLss;
	OrderedHitTriplets hitTriplets;
};


#endif
