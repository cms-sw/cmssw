// Original author: Leonardo Cristella

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"
#include "TSToSimTSHitLCAssociatorByEnergyScoreImpl.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

class TSToSimTSHitLCAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
public:
  explicit TSToSimTSHitLCAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
  ~TSToSimTSHitLCAssociatorByEnergyScoreProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> rhtools_;
  std::vector<edm::InputTag> hits_label_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hits_token_;
};

TSToSimTSHitLCAssociatorByEnergyScoreProducer::TSToSimTSHitLCAssociatorByEnergyScoreProducer(const edm::ParameterSet &ps)
    : hitMap_(consumes<std::unordered_map<DetId, const unsigned int>>(ps.getParameter<edm::InputTag>("hitMapTag"))),
      caloGeometry_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      hardScatterOnly_(ps.getParameter<bool>("hardScatterOnly")),
      hits_label_(ps.getParameter<std::vector<edm::InputTag>>("hits")) {
  rhtools_.reset(new hgcal::RecHitTools());

  for (auto &label : hits_label_) {
    hits_token_.push_back(consumes<HGCRecHitCollection>(label));
  }

  // Register the product
  produces<ticl::TracksterToSimTracksterHitLCAssociator>();
}

TSToSimTSHitLCAssociatorByEnergyScoreProducer::~TSToSimTSHitLCAssociatorByEnergyScoreProducer() {}

void TSToSimTSHitLCAssociatorByEnergyScoreProducer::produce(edm::StreamID,
                                                            edm::Event &iEvent,
                                                            const edm::EventSetup &es) const {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeometry_);
  rhtools_->setGeometry(*geom);

  std::vector<const HGCRecHit *> hits;
  for (auto &token : hits_token_) {
    edm::Handle<HGCRecHitCollection> hits_handle;
    iEvent.getByToken(token, hits_handle);
    for (const auto &hit : *hits_handle) {
      hits.push_back(&hit);
    }
  }

  const auto hitMap = &iEvent.get(hitMap_);

  auto impl = std::make_unique<TSToSimTSHitLCAssociatorByEnergyScoreImpl>(
      iEvent.productGetter(), hardScatterOnly_, rhtools_, hitMap, hits);
  auto toPut = std::make_unique<ticl::TracksterToSimTracksterHitLCAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void TSToSimTSHitLCAssociatorByEnergyScoreProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hitMapTag", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
  desc.add<std::vector<edm::InputTag>>("hits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<bool>("hardScatterOnly", true);

  cfg.add("simTracksterHitLCAssociatorByEnergyScore", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TSToSimTSHitLCAssociatorByEnergyScoreProducer);
