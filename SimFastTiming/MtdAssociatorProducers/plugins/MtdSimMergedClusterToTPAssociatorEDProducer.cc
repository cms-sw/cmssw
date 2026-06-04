// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/Associations/interface/MtdSimMergedClusterToTPAssociator.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class declaration
//

class MtdSimMergedClusterToTPAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit MtdSimMergedClusterToTPAssociatorEDProducer(const edm::ParameterSet &);
  ~MtdSimMergedClusterToTPAssociatorEDProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<MtdSimMergedClusterCollection> simMergedClustersToken_;
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<reco::MtdSimMergedClusterToTPAssociator> associatorToken_;
};

MtdSimMergedClusterToTPAssociatorEDProducer::MtdSimMergedClusterToTPAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<reco::MergedSimToTPCollectionMtd>();
  produces<reco::TPToMergedSimCollectionMtd>();

  simMergedClustersToken_ =
      consumes<MtdSimMergedClusterCollection>(pset.getParameter<edm::InputTag>("mtdSimMergedClustersTag"));
  tpToken_ = consumes<TrackingParticleCollection>(pset.getParameter<edm::InputTag>("trackingParticlesTag"));
  associatorToken_ = consumes<reco::MtdSimMergedClusterToTPAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

MtdSimMergedClusterToTPAssociatorEDProducer::~MtdSimMergedClusterToTPAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MtdSimMergedClusterToTPAssociatorEDProducer::produce(edm::StreamID,
                                                          edm::Event &iEvent,
                                                          const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<reco::MtdSimMergedClusterToTPAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  edm::Handle<MtdSimMergedClusterCollection> simMergedClusters;
  iEvent.getByToken(simMergedClustersToken_, simMergedClusters);

  edm::Handle<TrackingParticleCollection> trackingParticles;
  iEvent.getByToken(tpToken_, trackingParticles);

  reco::MergedSimToTPCollectionMtd simToTPColl = theAssociator->associateSimToTP(simMergedClusters, trackingParticles);
  reco::TPToMergedSimCollectionMtd tpToSimColl = theAssociator->associateTPToSim(simMergedClusters, trackingParticles);

  auto s2tp = std::make_unique<reco::MergedSimToTPCollectionMtd>(simToTPColl);
  auto tp2s = std::make_unique<reco::TPToMergedSimCollectionMtd>(tpToSimColl);

  iEvent.put(std::move(s2tp));
  iEvent.put(std::move(tp2s));
}

void MtdSimMergedClusterToTPAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("associator", edm::InputTag("mtdSimMergedClusterToTPAssociatorByTrackId"));
  desc.add<edm::InputTag>("mtdSimMergedClustersTag", edm::InputTag("mtdSimMergedClusterProducer", ""));
  desc.add<edm::InputTag>("trackingParticlesTag", edm::InputTag("mix", "MergedTrackTruth"));

  cfg.add("mtdSimMergedClusterToTPAssociationDefault", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MtdSimMergedClusterToTPAssociatorEDProducer);
