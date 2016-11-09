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

class HitPairEDProducer: public edm::stream::EDProducer<> {
public:
  HitPairEDProducer(const edm::ParameterSet& iConfig);
  ~HitPairEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> seedingLayerToken_;
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;
  edm::EDGetTokenT<bool> clusterCheckToken_;

  edm::RunningAverage localRA_;
  LayerHitMapCache layerCache_;
  const unsigned int maxElement_;

  HitPairGeneratorFromLayerPair generator_;
  std::vector<unsigned> layerPairBegins_;

  const bool produceSeedingHitSets_;
  const bool produceIntermediateHitDoublets_;
};


HitPairEDProducer::HitPairEDProducer(const edm::ParameterSet& iConfig):
  seedingLayerToken_(consumes<SeedingLayerSetsHits>(iConfig.getParameter<edm::InputTag>("seedingLayers"))),
  regionToken_(consumes<edm::OwnVector<TrackingRegion> >(iConfig.getParameter<edm::InputTag>("trackingRegions"))),
  clusterCheckToken_(consumes<bool>(iConfig.getParameter<edm::InputTag>("clusterCheck"))),
  maxElement_(iConfig.getParameter<unsigned int>("maxElement")),
  generator_(0, 1, nullptr, maxElement_), // these indices are dummy, TODO: cleanup HitPairGeneratorFromLayerPair
  layerPairBegins_(iConfig.getParameter<std::vector<unsigned> >("layerPairs")),
  produceSeedingHitSets_(iConfig.getParameter<bool>("produceSeedingHitSets")),
  produceIntermediateHitDoublets_(iConfig.getParameter<bool>("produceIntermediateHitDoublets"))
{
  if(!produceIntermediateHitDoublets_ && !produceSeedingHitSets_)
    throw cms::Exception("Configuration") << "HitPairEDProducer requires either produceIntermediateHitDoublets or produceSeedingHitSets to be True. If neither are needed, just remove this module from your sequence/path as it doesn't do anything useful";

  if(layerPairBegins_.empty())
    throw cms::Exception("Configuration") << "HitPairEDProducer requires at least index for layer pairs (layerPairs parameter), none was given";

  if(produceSeedingHitSets_)
    produces<RegionsSeedingHitSets>();
  if(produceIntermediateHitDoublets_)
    produces<IntermediateHitDoublets>();
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
  edm::Handle<bool> hclusterCheck;
  iEvent.getByToken(clusterCheckToken_, hclusterCheck);
  const bool clusterCheckOk = *hclusterCheck;

  edm::Handle<SeedingLayerSetsHits> hlayers;
  iEvent.getByToken(seedingLayerToken_, hlayers);
  const auto& layers = *hlayers;
  if(layers.numberOfLayersInSet() < 2)
    throw cms::Exception("LogicError") << "HitPairEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 2, got " << layers.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, or SeedingLayersEDProducer.";

  edm::Handle<edm::OwnVector<TrackingRegion> > hregions;
  iEvent.getByToken(regionToken_, hregions);
  const auto& regions = *hregions;

  std::unique_ptr<RegionsSeedingHitSets> seedingHitSets;
  std::unique_ptr<IntermediateHitDoublets> intermediateHitDoublets;
  if(produceSeedingHitSets_)
    seedingHitSets = std::make_unique<RegionsSeedingHitSets>();
  if(produceIntermediateHitDoublets_)
    intermediateHitDoublets = std::make_unique<IntermediateHitDoublets>(&layers);

  if(!clusterCheckOk) {
    if(produceSeedingHitSets_)
      iEvent.put(std::move(seedingHitSets));
    if(produceIntermediateHitDoublets_)
      iEvent.put(std::move(intermediateHitDoublets));
    return;
  }

  if(produceSeedingHitSets_)
    seedingHitSets->reserve(regions.size(), localRA_.upper());
  if(produceIntermediateHitDoublets_)
    intermediateHitDoublets->reserve(regions.size(), layers.size());


  LogDebug("HitPairEDProducer") << "Creating doublets for " << regions.size() << " and " << layers.size() << " layer sets";

  std::vector<SeedingLayerSetsHits::SeedingLayerSet> layerPairs;
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

  LayerHitMapCache hitCacheTmp; // used if !produceIntermediateHitDoublets_
  for(const TrackingRegion& region: regions) {
    auto seedingHitSetsFiller = RegionsSeedingHitSets::dummyFiller();
    auto intermediateHitDoubletsFiller = IntermediateHitDoublets::dummyFiller();
    auto hitCachePtr = &hitCacheTmp;
    if(produceSeedingHitSets_) {
      seedingHitSetsFiller = seedingHitSets->beginRegion(&region);
    }
    if(produceIntermediateHitDoublets_) {
      intermediateHitDoubletsFiller = intermediateHitDoublets->beginRegion(&region);
      hitCachePtr = &(intermediateHitDoubletsFiller.layerHitMapCache());
    }
    else {
      hitCacheTmp.clear();
    }


    for(SeedingLayerSetsHits::SeedingLayerSet layerSet: layerPairs) {
      auto doublets = generator_.doublets(region, iEvent, iSetup, layerSet, *hitCachePtr);
      LogTrace("HitPairEDProducer") << " created " << doublets.size() << " doublets for layers " << layerSet[0].index() << "," << layerSet[1].index();
      if(doublets.empty()) continue; // don't bother if no pairs from these layers
      if(produceSeedingHitSets_) {
        for(size_t i=0, size=doublets.size(); i<size; ++i) {
          seedingHitSetsFiller.emplace_back(doublets.hit(i, HitDoublets::inner),
                                            doublets.hit(i, HitDoublets::outer));
        }
      }
      if(produceIntermediateHitDoublets_) {
        intermediateHitDoubletsFiller.addDoublets(layerSet, std::move(doublets));
      }
    }
  }

  if(produceSeedingHitSets_) {
    seedingHitSets->shrink_to_fit();
    localRA_.update(seedingHitSets->size());
    iEvent.put(std::move(seedingHitSets));
  }
  if(produceIntermediateHitDoublets_) {
    intermediateHitDoublets->shrink_to_fit();
    iEvent.put(std::move(intermediateHitDoublets));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HitPairEDProducer);
