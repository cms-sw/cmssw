#include "SeedCombiner.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"

using namespace edm;
//using namespace reco;

SeedCombiner::SeedCombiner(
    const edm::ParameterSet& cfg) 
  : 
    inputCollections_(cfg.getParameter<std::vector<edm::InputTag> >("seedCollections"))
{
    produces<TrajectorySeedCollection>();
}


SeedCombiner::~SeedCombiner()
{
}


void SeedCombiner::beginRun(edm::Run & run, const edm::EventSetup& es)
{
}

void SeedCombiner::produce(edm::Event& ev, const edm::EventSetup& es)
{
    // Read inputs, and count total seeds
    size_t ninputs = inputCollections_.size();
    size_t nseeds = 0;
    std::vector<Handle<TrajectorySeedCollection > > seedCollections(ninputs);
    for (size_t i = 0; i < ninputs; ++i) {
        ev.getByLabel(inputCollections_[i], seedCollections[i]);
        nseeds += seedCollections[i]->size();
    }

    // Prepare output collections, with the correct capacity
    std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
    result->reserve( nseeds );

    // Write into output collection
    for (std::vector<Handle<TrajectorySeedCollection > >::const_iterator 
            it = seedCollections.begin(); 
            it != seedCollections.end(); 
            ++it) {
        result->insert(result->end(), (*it)->begin(), (*it)->end());
    }

    // Save result into the event
    ev.put(result);
}
