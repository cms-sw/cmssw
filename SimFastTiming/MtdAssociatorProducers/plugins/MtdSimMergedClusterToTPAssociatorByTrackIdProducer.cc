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

#include "MtdSimMergedClusterToTPAssociatorByTrackIdImpl.h"

//
// Class declaration
//

class MtdSimMergedClusterToTPAssociatorByTrackIdProducer : public edm::global::EDProducer<> {
public:
  explicit MtdSimMergedClusterToTPAssociatorByTrackIdProducer(const edm::ParameterSet &);
  ~MtdSimMergedClusterToTPAssociatorByTrackIdProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<reco::SimToTPCollectionMtd> simToTPMapToken_;
  edm::EDGetTokenT<reco::TPToSimCollectionMtd> tpToSimMapToken_;
};

MtdSimMergedClusterToTPAssociatorByTrackIdProducer::MtdSimMergedClusterToTPAssociatorByTrackIdProducer(
    const edm::ParameterSet &pset) {
  // Register the product
  produces<reco::MtdSimMergedClusterToTPAssociator>();

  simToTPMapToken_ = consumes<reco::SimToTPCollectionMtd>(pset.getParameter<edm::InputTag>("simToTPMap"));
  tpToSimMapToken_ = consumes<reco::TPToSimCollectionMtd>(pset.getParameter<edm::InputTag>("tpToSimMap"));
}

MtdSimMergedClusterToTPAssociatorByTrackIdProducer::~MtdSimMergedClusterToTPAssociatorByTrackIdProducer() {}

void MtdSimMergedClusterToTPAssociatorByTrackIdProducer::produce(edm::StreamID,
                                                                 edm::Event &iEvent,
                                                                 const edm::EventSetup &es) const {
  auto simToTPMapHandle = iEvent.getHandle(simToTPMapToken_);
  reco::SimToTPCollectionMtd simToTPMap = *simToTPMapHandle;
  auto tpToSimMapHandle = iEvent.getHandle(tpToSimMapToken_);
  reco::TPToSimCollectionMtd tpToSimMap = *tpToSimMapHandle;

  auto impl =
      std::make_unique<MtdSimMergedClusterToTPAssociatorByTrackIdImpl>(iEvent.productGetter(), simToTPMap, tpToSimMap);
  auto toPut = std::make_unique<reco::MtdSimMergedClusterToTPAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

void MtdSimMergedClusterToTPAssociatorByTrackIdProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("simToTPMap", edm::InputTag("mtdSimLayerClusterToTPAssociation"));
  desc.add<edm::InputTag>("tpToSimMap", edm::InputTag("mtdSimLayerClusterToTPAssociation"));

  cfg.add("mtdSimMergedClusterToTPAssociatorByTrackId", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MtdSimMergedClusterToTPAssociatorByTrackIdProducer);
