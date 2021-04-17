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

#include "TrackingTools/PatternTools/interface/ClusterRemovalRefSetter.h"
#include "FWCore/Utilities/interface/transform.h"

using namespace edm;
//using namespace reco;

SeedCombiner::SeedCombiner(const edm::ParameterSet& cfg) {
  inputCollections_ =
      edm::vector_transform(cfg.getParameter<std::vector<edm::InputTag> >("seedCollections"),
                            [this](edm::InputTag const& tag) { return consumes<TrajectorySeedCollection>(tag); });
  produces<TrajectorySeedCollection>();
  reKeing_ = false;
  if (cfg.exists("clusterRemovalInfos")) {
    clusterRemovalInfos_ = cfg.getParameter<std::vector<edm::InputTag> >("clusterRemovalInfos");
    clusterRemovalTokens_.resize(clusterRemovalInfos_.size());
    for (unsigned int i = 0; i < clusterRemovalInfos_.size(); ++i)
      if (!(clusterRemovalInfos_[i] == edm::InputTag("")))
        clusterRemovalTokens_[i] = consumes<reco::ClusterRemovalInfo>(clusterRemovalInfos_[i]);
    if (!clusterRemovalInfos_.empty() && clusterRemovalInfos_.size() == inputCollections_.size())
      reKeing_ = true;
  }
}

SeedCombiner::~SeedCombiner() {}

void SeedCombiner::produce(edm::Event& ev, const edm::EventSetup& es) {
  // Read inputs, and count total seeds
  size_t ninputs = inputCollections_.size();
  size_t nseeds = 0;
  std::vector<Handle<TrajectorySeedCollection> > seedCollections(ninputs);
  for (size_t i = 0; i < ninputs; ++i) {
    ev.getByToken(inputCollections_[i], seedCollections[i]);
    nseeds += seedCollections[i]->size();
  }

  // Prepare output collections, with the correct capacity
  auto result = std::make_unique<TrajectorySeedCollection>();
  result->reserve(nseeds);

  // Write into output collection
  unsigned int iSC = 0, iSC_max = seedCollections.size();
  for (; iSC != iSC_max; ++iSC) {
    Handle<TrajectorySeedCollection>& collection = seedCollections[iSC];
    if (reKeing_ && !(clusterRemovalInfos_[iSC] == edm::InputTag(""))) {
      ClusterRemovalRefSetter refSetter(ev, clusterRemovalTokens_[iSC]);

      for (TrajectorySeedCollection::const_iterator iS = collection->begin(); iS != collection->end(); ++iS) {
        TrajectorySeed::RecHitContainer newRecHitContainer;
        newRecHitContainer.reserve(iS->nHits());
        //loop seed rechits, copy over and rekey.
        for (auto const& recHit : iS->recHits()) {
          newRecHitContainer.push_back(recHit);
          refSetter.reKey(&newRecHitContainer.back());
        }
        result->push_back(TrajectorySeed(iS->startingState(), std::move(newRecHitContainer), iS->direction()));
      }
    } else {
      //just insert the new seeds as they are
      result->insert(result->end(), collection->begin(), collection->end());
    }
  }

  // Save result into the event
  ev.put(std::move(result));
}
