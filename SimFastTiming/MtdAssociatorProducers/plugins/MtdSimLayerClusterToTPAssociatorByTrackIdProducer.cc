// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "MtdSimLayerClusterToTPAssociatorByTrackIdImpl.h"

//
// Class declaration
//

class MtdSimLayerClusterToTPAssociatorByTrackIdProducer : public edm::global::EDProducer<> {
public:
  explicit MtdSimLayerClusterToTPAssociatorByTrackIdProducer(const edm::ParameterSet &);
  ~MtdSimLayerClusterToTPAssociatorByTrackIdProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
};

MtdSimLayerClusterToTPAssociatorByTrackIdProducer::MtdSimLayerClusterToTPAssociatorByTrackIdProducer(
    const edm::ParameterSet &pset) {
  // Register the product
  produces<reco::MtdSimLayerClusterToTPAssociator>();
}

MtdSimLayerClusterToTPAssociatorByTrackIdProducer::~MtdSimLayerClusterToTPAssociatorByTrackIdProducer() {}

void MtdSimLayerClusterToTPAssociatorByTrackIdProducer::produce(edm::StreamID,
                                                                edm::Event &iEvent,
                                                                const edm::EventSetup &es) const {
  auto impl = std::make_unique<MtdSimLayerClusterToTPAssociatorByTrackIdImpl>(iEvent.productGetter());
  auto toPut = std::make_unique<reco::MtdSimLayerClusterToTPAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void MtdSimLayerClusterToTPAssociatorByTrackIdProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;

  cfg.add("mtdSimLayerClusterToTPAssociatorByTrackId", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MtdSimLayerClusterToTPAssociatorByTrackIdProducer);
