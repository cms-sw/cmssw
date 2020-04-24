#ifndef RecoTracker_TkSeedGenerator_SeedCreatorFromRegionHitsEDProducerT_H
#define RecoTracker_TkSeedGenerator_SeedCreatorFromRegionHitsEDProducerT_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

template <typename T_SeedCreator>
class SeedCreatorFromRegionHitsEDProducerT: public edm::stream::EDProducer<> {
public:

  SeedCreatorFromRegionHitsEDProducerT(const edm::ParameterSet& iConfig);
  ~SeedCreatorFromRegionHitsEDProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<RegionsSeedingHitSets> seedingHitSetsToken_;
  T_SeedCreator seedCreator_;
  std::unique_ptr<SeedComparitor> comparitor_;
};

template <typename T_SeedCreator>
SeedCreatorFromRegionHitsEDProducerT<T_SeedCreator>::SeedCreatorFromRegionHitsEDProducerT(const edm::ParameterSet& iConfig):
  seedingHitSetsToken_(consumes<RegionsSeedingHitSets>(iConfig.getParameter<edm::InputTag>("seedingHitSets"))),
  seedCreator_(iConfig)
{
  edm::ConsumesCollector iC = consumesCollector();
  edm::ParameterSet comparitorPSet = iConfig.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  comparitor_.reset((comparitorName == "none") ? nullptr : SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC));

  produces<TrajectorySeedCollection>();
}

template <typename T_SeedCreator>
void SeedCreatorFromRegionHitsEDProducerT<T_SeedCreator>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("seedingHitSets", edm::InputTag("hitPairEDProducer"));
  T_SeedCreator::fillDescriptions(desc);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything(); // until we have moved SeedComparitor too to EDProducers
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);

  auto label = std::string("seedCreatorFromRegion") + T_SeedCreator::fillDescriptionsLabel() + "EDProducer";
  descriptions.add(label, desc);
}

template <typename T_SeedCreator>
void SeedCreatorFromRegionHitsEDProducerT<T_SeedCreator>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<RegionsSeedingHitSets> hseedingHitSets;
  iEvent.getByToken(seedingHitSetsToken_, hseedingHitSets);
  const auto& seedingHitSets = *hseedingHitSets;

  auto seeds = std::make_unique<TrajectorySeedCollection>();
  seeds->reserve(seedingHitSets.size());

  if(comparitor_)
    comparitor_->init(iEvent, iSetup);

  for(const auto& regionSeedingHitSets: seedingHitSets) {
    const TrackingRegion& region = regionSeedingHitSets.region();
    seedCreator_.init(region, iSetup, comparitor_.get());

    for(const SeedingHitSet& hits: regionSeedingHitSets) {
      // TODO: do we really need a comparitor at this point? It is
      // used in triplet and quadruplet generators, as well as inside
      // seedCreator.
      if(!comparitor_ || comparitor_->compatible(hits)) {
        seedCreator_.makeSeed(*seeds, hits);
      }
    }
  }

  seeds->shrink_to_fit();
  iEvent.put(std::move(seeds));
}

#endif
