#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

namespace { class ImplBase; }

class HitPairEDProducer: public edm::stream::EDProducer<> {
public:
  HitPairEDProducer(const edm::ParameterSet& iConfig);
  ~HitPairEDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> seedingLayerToken_;
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;
  edm::EDGetTokenT<bool> clusterCheckToken_;

  std::unique_ptr<::ImplBase> impl_;
};

namespace {
  class ImplBase {
  public:
    ImplBase(const edm::ParameterSet& iConfig);
    virtual ~ImplBase() = default;

    virtual void produces(edm::ProducerBase& producer) const = 0;

    virtual void produce(const SeedingLayerSetsHits& layers, const edm::OwnVector<TrackingRegion>& regions, const bool clusterCheckOk,
                         edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  protected:
    void makeLayerPairs(const SeedingLayerSetsHits& layers, std::vector<SeedingLayerSetsHits::SeedingLayerSet>& layerPairs);

    edm::RunningAverage localRA_;
    const unsigned int maxElement_;

    HitPairGeneratorFromLayerPair generator_;
    std::vector<unsigned> layerPairBegins_;
  };
  ImplBase::ImplBase(const edm::ParameterSet& iConfig):
    maxElement_(iConfig.getParameter<unsigned int>("maxElement")),
    generator_(0, 1, nullptr, maxElement_), // these indices are dummy, TODO: cleanup HitPairGeneratorFromLayerPair
    layerPairBegins_(iConfig.getParameter<std::vector<unsigned> >("layerPairs"))
  {
    if(layerPairBegins_.empty())
      throw cms::Exception("Configuration") << "HitPairEDProducer requires at least index for layer pairs (layerPairs parameter), none was given";
  }
  void ImplBase::makeLayerPairs(const SeedingLayerSetsHits& layers, std::vector<SeedingLayerSetsHits::SeedingLayerSet>& layerPairs) {
    if(layers.numberOfLayersInSet() > 2) {
      for(const auto& layerSet: layers) {
        for(const auto pairBeginIndex: layerPairBegins_) {
          if(pairBeginIndex+1 >= layers.numberOfLayersInSet()) {
            throw cms::Exception("LogicError") << "Layer pair index " << pairBeginIndex << " is out of bounds, input SeedingLayerSetsHits has only " << layers.numberOfLayersInSet() << " layers per set, and the index+1 must be < than the number of layers in set";
          }

          // Take only the requested pair of the set
          SeedingLayerSetsHits::SeedingLayerSet pairCandidate = layerSet.slice(pairBeginIndex, pairBeginIndex+1);

          // it would be trivial to use 128-bit bitfield for O(1) check
          // if a layer pair has been inserted, but let's test first how
          // a "straightforward" solution works
          auto found = std::find_if(layerPairs.begin(), layerPairs.end(), [&](const SeedingLayerSetsHits::SeedingLayerSet& pair) {
              return pair[0].index() == pairCandidate[0].index() && pair[1].index() == pairCandidate[1].index();
            });
          if(found != layerPairs.end())
            continue;

          layerPairs.push_back(pairCandidate);
        }
      }
    }
    else {
      if(layerPairBegins_.size() != 1) {
        throw cms::Exception("LogicError") << "With pairs of input layers, it doesn't make sense to specify more than one input layer pair, got " << layerPairBegins_.size();
      }
      if(layerPairBegins_[0] != 0) {
        throw cms::Exception("LogicError") << "With pairs of input layers, it doesn't make sense to specify other input layer pair than 0; got " << layerPairBegins_[0];
      }

      layerPairs.reserve(layers.size());
      for(const auto& set: layers)
        layerPairs.push_back(set);
    }
  }


  /////
  template <typename T_SeedingHitSets, typename T_IntermediateHitDoublets>
  struct Impl: public ImplBase {
    Impl(const edm::ParameterSet& iConfig): ImplBase(iConfig) {}
    ~Impl() override = default;

    void produces(edm::ProducerBase& producer) const override {
      T_SeedingHitSets::produces(producer);
      T_IntermediateHitDoublets::produces(producer);
    }

    void produce(const SeedingLayerSetsHits& layers, const edm::OwnVector<TrackingRegion>& regions, const bool clusterCheckOk,
                 edm::Event& iEvent, const edm::EventSetup& iSetup) override {
      auto seedingHitSetsProducer = T_SeedingHitSets(&localRA_);
      auto intermediateHitDoubletsProducer = T_IntermediateHitDoublets(&layers);

      if(!clusterCheckOk) {
        seedingHitSetsProducer.putEmpty(iEvent);
        intermediateHitDoubletsProducer.putEmpty(iEvent);
        return;
      }

      seedingHitSetsProducer.reserve(regions.size());
      intermediateHitDoubletsProducer.reserve(regions.size());

      std::vector<SeedingLayerSetsHits::SeedingLayerSet> layerPairs;
      makeLayerPairs(layers, layerPairs);

      for(const TrackingRegion& region: regions) {
        auto hitCachePtr_filler_shs = seedingHitSetsProducer.beginRegion(&region, nullptr);
        auto hitCachePtr_filler_ihd = intermediateHitDoubletsProducer.beginRegion(&region, std::get<0>(hitCachePtr_filler_shs));
        auto hitCachePtr = std::get<0>(hitCachePtr_filler_ihd);

        for(SeedingLayerSetsHits::SeedingLayerSet layerSet: layerPairs) {
          auto doublets = generator_.doublets(region, iEvent, iSetup, layerSet, *hitCachePtr);
          LogTrace("HitPairEDProducer") << " created " << doublets.size() << " doublets for layers " << layerSet[0].index() << "," << layerSet[1].index();
          if(doublets.empty()) continue; // don't bother if no pairs from these layers
          seedingHitSetsProducer.fill(std::get<1>(hitCachePtr_filler_shs), doublets);
          intermediateHitDoubletsProducer.fill(std::get<1>(hitCachePtr_filler_ihd), layerSet, std::move(doublets));
        }
      }

      seedingHitSetsProducer.put(iEvent);
      intermediateHitDoubletsProducer.put(iEvent);
    }
  };

  /////
  class DoNothing {
  public:
    DoNothing(const SeedingLayerSetsHits *) {}
    DoNothing(edm::RunningAverage *) {}

    static void produces(edm::ProducerBase&) {};

    void reserve(size_t) {}

    auto beginRegion(const TrackingRegion *, LayerHitMapCache *ptr) {
      return std::make_tuple(ptr, 0);
    }

    void fill(int, const HitDoublets&) {}
    void fill(int, const SeedingLayerSetsHits::SeedingLayerSet&, HitDoublets&&) {}

    void put(edm::Event&) {}
    void putEmpty(edm::Event&) {}
  };

  /////
  class ImplSeedingHitSets {
  public:
    ImplSeedingHitSets(edm::RunningAverage *localRA):
      seedingHitSets_(std::make_unique<RegionsSeedingHitSets>()),
      localRA_(localRA)
    {}

    static void produces(edm::ProducerBase& producer) {
      producer.produces<RegionsSeedingHitSets>();
    }

    void reserve(size_t regionsSize) {
      seedingHitSets_->reserve(regionsSize, localRA_->upper());
    }

    auto beginRegion(const TrackingRegion *region, LayerHitMapCache *) {
      hitCacheTmp_.clear();
      return std::make_tuple(&hitCacheTmp_, seedingHitSets_->beginRegion(region));
    }

    void fill(RegionsSeedingHitSets::RegionFiller& filler, const HitDoublets& doublets) {
      for(size_t i=0, size=doublets.size(); i<size; ++i) {
        filler.emplace_back(doublets.hit(i, HitDoublets::inner),
                            doublets.hit(i, HitDoublets::outer));
      }
    }

    void put(edm::Event& iEvent) {
      seedingHitSets_->shrink_to_fit();
      localRA_->update(seedingHitSets_->size());
      putEmpty(iEvent);
    }

    void putEmpty(edm::Event& iEvent) {
      iEvent.put(std::move(seedingHitSets_));
    }

  private:
    std::unique_ptr<RegionsSeedingHitSets> seedingHitSets_;
    edm::RunningAverage *localRA_;
    LayerHitMapCache hitCacheTmp_; // used if !produceIntermediateHitDoublets
  };

  /////
  class ImplIntermediateHitDoublets {
  public:
    ImplIntermediateHitDoublets(const SeedingLayerSetsHits *layers):
      intermediateHitDoublets_(std::make_unique<IntermediateHitDoublets>(layers)),
      layers_(layers)
    {}

    static void produces(edm::ProducerBase& producer) {
      producer.produces<IntermediateHitDoublets>();
    }

    void reserve(size_t regionsSize) {
      intermediateHitDoublets_->reserve(regionsSize, layers_->size());
    }

    auto beginRegion(const TrackingRegion *region, LayerHitMapCache *) {
      auto filler = intermediateHitDoublets_->beginRegion(region);
      return std::make_tuple(&(filler.layerHitMapCache()), std::move(filler));
    }

    void fill(IntermediateHitDoublets::RegionFiller& filler, const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets) {
      filler.addDoublets(layerSet, std::move(doublets));
    }

    void put(edm::Event& iEvent) {
      intermediateHitDoublets_->shrink_to_fit();
      putEmpty(iEvent);
    }

    void putEmpty(edm::Event& iEvent) {
      iEvent.put(std::move(intermediateHitDoublets_));
    }

  private:
    std::unique_ptr<IntermediateHitDoublets> intermediateHitDoublets_;
    const SeedingLayerSetsHits *layers_;
  };
}



HitPairEDProducer::HitPairEDProducer(const edm::ParameterSet& iConfig):
  seedingLayerToken_(consumes<SeedingLayerSetsHits>(iConfig.getParameter<edm::InputTag>("seedingLayers"))),
  regionToken_(consumes<edm::OwnVector<TrackingRegion> >(iConfig.getParameter<edm::InputTag>("trackingRegions")))
{
  const bool produceSeedingHitSets = iConfig.getParameter<bool>("produceSeedingHitSets");
  const bool produceIntermediateHitDoublets = iConfig.getParameter<bool>("produceIntermediateHitDoublets");

  if(produceSeedingHitSets && produceIntermediateHitDoublets)
    impl_ = std::make_unique<::Impl<::ImplSeedingHitSets, ::ImplIntermediateHitDoublets>>(iConfig);
  else if(produceSeedingHitSets)
    impl_ = std::make_unique<::Impl<::ImplSeedingHitSets, ::DoNothing>>(iConfig);
  else if(produceIntermediateHitDoublets)
    impl_ = std::make_unique<::Impl<::DoNothing, ::ImplIntermediateHitDoublets>>(iConfig);
  else
    throw cms::Exception("Configuration") << "HitPairEDProducer requires either produceIntermediateHitDoublets or produceSeedingHitSets to be True. If neither are needed, just remove this module from your sequence/path as it doesn't do anything useful";

  auto clusterCheckTag = iConfig.getParameter<edm::InputTag>("clusterCheck");
  if(clusterCheckTag.label() != "")
    clusterCheckToken_ = consumes<bool>(clusterCheckTag);

  impl_->produces(*this);
}

void HitPairEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("seedingLayers", edm::InputTag("seedingLayersEDProducer"));
  desc.add<edm::InputTag>("trackingRegions", edm::InputTag("globalTrackingRegionFromBeamSpot"));
  desc.add<edm::InputTag>("clusterCheck", edm::InputTag("trackerClusterCheck"));
  desc.add<bool>("produceSeedingHitSets", false);
  desc.add<bool>("produceIntermediateHitDoublets", false);
  desc.add<unsigned int>("maxElement", 1000000);
  desc.add<std::vector<unsigned> >("layerPairs", std::vector<unsigned>{{0}})->setComment("Indices to the pairs of consecutive layers, i.e. 0 means (0,1), 1 (1,2) etc.");

  descriptions.add("hitPairEDProducerDefault", desc);
}

void HitPairEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool clusterCheckOk = true;
  if(!clusterCheckToken_.isUninitialized()) {
    edm::Handle<bool> hclusterCheck;
    iEvent.getByToken(clusterCheckToken_, hclusterCheck);
    clusterCheckOk = *hclusterCheck;
  }

  edm::Handle<SeedingLayerSetsHits> hlayers;
  iEvent.getByToken(seedingLayerToken_, hlayers);
  const auto& layers = *hlayers;
  if(layers.numberOfLayersInSet() < 2)
    throw cms::Exception("LogicError") << "HitPairEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 2, got " << layers.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, or SeedingLayersEDProducer.";

  edm::Handle<edm::OwnVector<TrackingRegion> > hregions;
  iEvent.getByToken(regionToken_, hregions);

  impl_->produce(layers, *hregions, clusterCheckOk, iEvent, iSetup);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HitPairEDProducer);
