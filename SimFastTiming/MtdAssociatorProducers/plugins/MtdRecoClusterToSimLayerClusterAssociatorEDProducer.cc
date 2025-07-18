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
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociator.h"

//
// class decleration
//

class MtdRecoClusterToSimLayerClusterAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit MtdRecoClusterToSimLayerClusterAssociatorEDProducer(const edm::ParameterSet &);
  ~MtdRecoClusterToSimLayerClusterAssociatorEDProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<FTLClusterCollection> btlRecoClustersToken_;
  edm::EDGetTokenT<FTLClusterCollection> etlRecoClustersToken_;
  edm::EDGetTokenT<MtdSimLayerClusterCollection> simClustersToken_;

  edm::EDGetTokenT<reco::MtdRecoClusterToSimLayerClusterAssociator> associatorToken_;
};

MtdRecoClusterToSimLayerClusterAssociatorEDProducer::MtdRecoClusterToSimLayerClusterAssociatorEDProducer(
    const edm::ParameterSet &pset) {
  produces<reco::SimToRecoCollectionMtd>();
  produces<reco::RecoToSimCollectionMtd>();

  btlRecoClustersToken_ = consumes<FTLClusterCollection>(pset.getParameter<edm::InputTag>("btlRecoClustersTag"));
  etlRecoClustersToken_ = consumes<FTLClusterCollection>(pset.getParameter<edm::InputTag>("etlRecoClustersTag"));
  simClustersToken_ = consumes<MtdSimLayerClusterCollection>(pset.getParameter<edm::InputTag>("mtdSimClustersTag"));
  associatorToken_ =
      consumes<reco::MtdRecoClusterToSimLayerClusterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

MtdRecoClusterToSimLayerClusterAssociatorEDProducer::~MtdRecoClusterToSimLayerClusterAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MtdRecoClusterToSimLayerClusterAssociatorEDProducer::produce(edm::StreamID,
                                                                  edm::Event &iEvent,
                                                                  const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<reco::MtdRecoClusterToSimLayerClusterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  edm::Handle<FTLClusterCollection> btlRecoClusters;
  iEvent.getByToken(btlRecoClustersToken_, btlRecoClusters);

  edm::Handle<FTLClusterCollection> etlRecoClusters;
  iEvent.getByToken(etlRecoClustersToken_, etlRecoClusters);

  edm::Handle<MtdSimLayerClusterCollection> simClusters;
  iEvent.getByToken(simClustersToken_, simClusters);

  // associate reco clus to sim layer clus
  reco::RecoToSimCollectionMtd recoToSimColl =
      theAssociator->associateRecoToSim(btlRecoClusters, etlRecoClusters, simClusters);
  reco::SimToRecoCollectionMtd simToRecoColl =
      theAssociator->associateSimToReco(btlRecoClusters, etlRecoClusters, simClusters);

  auto r2s = std::make_unique<reco::RecoToSimCollectionMtd>(recoToSimColl);
  auto s2r = std::make_unique<reco::SimToRecoCollectionMtd>(simToRecoColl);

  iEvent.put(std::move(r2s));
  iEvent.put(std::move(s2r));
}

void MtdRecoClusterToSimLayerClusterAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("associator", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociatorByHits"));
  desc.add<edm::InputTag>("mtdSimClustersTag", edm::InputTag("mix", "MergedMtdTruthLC"));
  desc.add<edm::InputTag>("btlRecoClustersTag", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("etlRecoClustersTag", edm::InputTag("mtdClusters", "FTLEndcap"));

  cfg.add("mtdRecoClusterToSimLayerClusterAssociationDefault", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MtdRecoClusterToSimLayerClusterAssociatorEDProducer);
