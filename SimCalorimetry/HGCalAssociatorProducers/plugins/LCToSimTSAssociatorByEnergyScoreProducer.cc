// Original author: Leonardo Cristella

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"
#include "LCToSimTSAssociatorByEnergyScoreImpl.h"

class LCToSimTSAssociatorByEnergyScoreProducer : public edm::global::EDProducer<> {
public:
  explicit LCToSimTSAssociatorByEnergyScoreProducer(const edm::ParameterSet &);
  ~LCToSimTSAssociatorByEnergyScoreProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  std::shared_ptr<hgcal::RecHitTools> rhtools_;
};

LCToSimTSAssociatorByEnergyScoreProducer::LCToSimTSAssociatorByEnergyScoreProducer(const edm::ParameterSet &ps)
    : caloGeometry_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {
  rhtools_.reset(new hgcal::RecHitTools());

  // Register the product
  produces<hgcal::LayerClusterToSimTracksterAssociator>();
}

LCToSimTSAssociatorByEnergyScoreProducer::~LCToSimTSAssociatorByEnergyScoreProducer() {}

void LCToSimTSAssociatorByEnergyScoreProducer::produce(edm::StreamID,
                                                       edm::Event &iEvent,
                                                       const edm::EventSetup &es) const {
  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeometry_);
  rhtools_->setGeometry(*geom);

  auto impl = std::make_unique<LCToSimTSAssociatorByEnergyScoreImpl>(iEvent.productGetter());
  auto toPut = std::make_unique<hgcal::LayerClusterToSimTracksterAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void LCToSimTSAssociatorByEnergyScoreProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  cfg.add("layerClusterSimTracksterAssociatorByEnergyScore", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LCToSimTSAssociatorByEnergyScoreProducer);
