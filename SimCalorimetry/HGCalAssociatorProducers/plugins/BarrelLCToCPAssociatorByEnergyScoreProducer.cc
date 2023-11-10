#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "BarrelLCToCPAssociatorByEnergyScoreImpl.h"


class BarrelLCToCPAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
  public:
    explicit BarrelLCToCPAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
    ~BarrelLCToCPAssociatorByEnergyScoreProducer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
    edm::EDGetTokenT<std::unordered_map<DetId, const reco::PFRecHit*>> hitMap_;
    const bool hardScatterOnly_;
};

BarrelLCToCPAssociatorByEnergyScoreProducer::BarrelLCToCPAssociatorByEnergyScoreProducer(const edm::ParameterSet &ps)
  : hitMap_(consumes<std::unordered_map<DetId, const reco::PFRecHit*>>(ps.getParameter<edm::InputTag>("hitMapTag"))),
    hardScatterOnly_(ps.getParameter<bool>("hardScatterOnly")) {

  produces<ticl::LayerClusterToCaloParticleAssociator>();
}

BarrelLCToCPAssociatorByEnergyScoreProducer::~BarrelLCToCPAssociatorByEnergyScoreProducer() {}

void BarrelLCToCPAssociatorByEnergyScoreProducer::produce(edm::StreamID,
							  edm::Event &iEvent,
							  const edm::EventSetup &es) const {

  const auto hitMap = &iEvent.get(hitMap_);

  auto impl =
    std::make_unique<BarrelLCToCPAssociatorByEnergyScoreImpl>(iEvent.productGetter(), hardScatterOnly_, hitMap);
  auto toPut = std::make_unique<ticl::LayerClusterToCaloParticleAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void BarrelLCToCPAssociatorByEnergyScoreProducer::fillDescriptions(edm::ConfigurationDescriptions& cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hitMapTag", edm::InputTag("barrelRecHitMapProducer"));
  desc.add<bool>("hardScatterOnly", true);

  cfg.add("barrelLayerClusterAssociatorByEnergyScore", desc);
}

DEFINE_FWK_MODULE(BarrelLCToCPAssociatorByEnergyScoreProducer);
