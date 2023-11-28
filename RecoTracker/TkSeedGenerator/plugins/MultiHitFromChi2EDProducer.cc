#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoTracker/PixelSeeding/interface/LayerTriplets.h"
#include "MultiHitGeneratorFromChi2.h"

class MultiHitFromChi2EDProducer : public edm::stream::EDProducer<> {
public:
  MultiHitFromChi2EDProducer(const edm::ParameterSet& iConfig);
  ~MultiHitFromChi2EDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  edm::RunningAverage localRA_;

  MultiHitGeneratorFromChi2 generator_;
};

MultiHitFromChi2EDProducer::MultiHitFromChi2EDProducer(const edm::ParameterSet& iConfig)
    : doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
      generator_(iConfig, consumesCollector()) {
  produces<RegionsSeedingHitSets>();
  produces<edm::OwnVector<BaseTrackerRecHit> >();
}

void MultiHitFromChi2EDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));

  MultiHitGeneratorFromChi2::fillDescriptions(desc);

  auto label = MultiHitGeneratorFromChi2::fillDescriptionsLabel() + std::string("EDProducerDefault");
  descriptions.add(label, desc);
}

void MultiHitFromChi2EDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if (seedingLayerHits.numberOfLayersInSet() < 3) {
    throw cms::Exception("LogicError")
        << "MultiHitFromChi2EDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 3, got "
        << seedingLayerHits.numberOfLayersInSet()
        << ". This is likely caused by a configuration error of this module, HitPairEDProducer, or "
           "SeedingLayersEDProducer.";
  }

  auto seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if (regionDoublets.empty()) {
    iEvent.put(std::move(seedingHitSets));
    return;
  }
  seedingHitSets->reserve(regionDoublets.regionSize(), localRA_.upper());
  generator_.initES(iSetup);

  // match-making of pair and triplet layers
  std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(seedingLayerHits);

  OrderedMultiHits multihits;
  multihits.reserve(localRA_.upper());
  std::vector<std::unique_ptr<BaseTrackerRecHit> > refittedHitStorage;
  refittedHitStorage.reserve(localRA_.upper() * 2);

  LogDebug("MultiHitFromChi2EDProducer") << "Creating multihits for " << regionDoublets.regionSize() << " regions, and "
                                         << trilayers.size() << " pair+3rd layers from "
                                         << regionDoublets.layerPairsSize() << " layer pairs";

  LayerHitMapCache hitCache;
  for (const auto& regionLayerPairs : regionDoublets) {
    const TrackingRegion& region = regionLayerPairs.region();

    auto seedingHitSetsFiller = RegionsSeedingHitSets::dummyFiller();
    seedingHitSetsFiller = seedingHitSets->beginRegion(&region);

    hitCache.clear();
    hitCache.extend(regionLayerPairs.layerHitMapCache());

    LogTrace("MultiHitFromChi2EDProducer") << " starting region";

    for (const auto& layerPair : regionLayerPairs) {
      LogTrace("MultiHitFromChi2EDProducer")
          << "  starting layer pair " << layerPair.innerLayerIndex() << "," << layerPair.outerLayerIndex();

      auto found = std::find_if(trilayers.begin(), trilayers.end(), [&](const LayerTriplets::LayerSetAndLayers& a) {
        return a.first[0].index() == layerPair.innerLayerIndex() && a.first[1].index() == layerPair.outerLayerIndex();
      });
      if (found == trilayers.end()) {
        auto exp = cms::Exception("LogicError") << "Did not find the layer pair from vector<pair+third layers>. This "
                                                   "is a sign of some internal inconsistency\n";
        exp << "I was looking for layer pair " << layerPair.innerLayerIndex() << "," << layerPair.outerLayerIndex()
            << ". Triplets have the following pairs:\n";
        for (const auto& a : trilayers) {
          exp << " " << a.first[0].index() << "," << a.first[1].index() << ": 3rd layers";
          for (const auto& b : a.second) {
            exp << " " << b.index();
          }
          exp << "\n";
        }
        throw exp;
      }
      const auto& thirdLayers = found->second;

      generator_.hitSets(region, multihits, layerPair.doublets(), thirdLayers, hitCache, refittedHitStorage);

#ifdef EDM_ML_DEBUG
      LogTrace("MultiHitFromChi2EDProducer")
          << "  created " << multihits.size() << " multihits for layer pair " << layerPair.innerLayerIndex() << ","
          << layerPair.outerLayerIndex() << " and 3rd layers";
      for (const auto& l : thirdLayers) {
        LogTrace("MultiHitFromChi2EDProducer") << "   " << l.index();
      }
#endif

      for (const SeedingHitSet& hitSet : multihits) {
        seedingHitSetsFiller.emplace_back(hitSet);
      }
      multihits.clear();
    }
  }
  localRA_.update(seedingHitSets->size());

  auto storage = std::make_unique<edm::OwnVector<BaseTrackerRecHit> >();
  storage->reserve(refittedHitStorage.size());
  for (auto& ptr : refittedHitStorage)
    storage->push_back(ptr.release());

  seedingHitSets->shrink_to_fit();
  storage->shrink_to_fit();
  iEvent.put(std::move(seedingHitSets));
  iEvent.put(std::move(storage));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MultiHitFromChi2EDProducer);
