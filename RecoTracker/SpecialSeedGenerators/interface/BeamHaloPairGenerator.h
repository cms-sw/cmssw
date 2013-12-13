#ifndef BeamHaloGenerators_BeamHaloPairGenerator_h
#define BeamHaloGenerators_BeamHaloPairGenerator_h
//FWK
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"

class BeamHaloPairGenerator : public OrderedHitsGenerator {
	public:
	BeamHaloPairGenerator(const edm::ParameterSet& conf);
	virtual ~BeamHaloPairGenerator(){};
	virtual const OrderedSeedingHits& run(const TrackingRegion& region, 
					      const edm::Event & ev, 
					      const edm::EventSetup& es);
	private:
	ctfseeding::SeedingLayerSets init(const edm::EventSetup& es);
	edm::ParameterSet conf_;
	OrderedHitPairs hitPairs;
	double theMaxTheta;
};


#endif
