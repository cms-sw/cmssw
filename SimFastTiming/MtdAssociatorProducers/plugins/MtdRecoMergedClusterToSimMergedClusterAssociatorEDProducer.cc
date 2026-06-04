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

#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociator.h"

//
// class decleration
//

class MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer(const edm::ParameterSet &);
  ~MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<FTLMergedClusterCollection> btlRecoClustersToken_;
  edm::EDGetTokenT<FTLMergedClusterCollection> etlRecoClustersToken_;
  edm::EDGetTokenT<MtdSimMergedClusterCollection> simMergedClustersToken_;

  edm::EDGetTokenT<reco::MtdRecoMergedClusterToSimMergedClusterAssociator> associatorToken_;
};

MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer::MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer(
    const edm::ParameterSet &pset) {
  produces<reco::MergedSimToRecoCollectionMtd>();
  produces<reco::MergedRecoToSimCollectionMtd>();

  btlRecoClustersToken_ = consumes<FTLMergedClusterCollection>(pset.getParameter<edm::InputTag>("btlRecoClustersTag"));
  etlRecoClustersToken_ = consumes<FTLMergedClusterCollection>(pset.getParameter<edm::InputTag>("etlRecoClustersTag"));
  simMergedClustersToken_ =
      consumes<MtdSimMergedClusterCollection>(pset.getParameter<edm::InputTag>("mtdSimMergedClustersTag"));
  associatorToken_ =
      consumes<reco::MtdRecoMergedClusterToSimMergedClusterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer::~MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer() {
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer::produce(edm::StreamID,
                                                                         edm::Event &iEvent,
                                                                         const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<reco::MtdRecoMergedClusterToSimMergedClusterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  edm::Handle<FTLMergedClusterCollection> btlRecoClusters;
  iEvent.getByToken(btlRecoClustersToken_, btlRecoClusters);

  edm::Handle<FTLMergedClusterCollection> etlRecoClusters;
  iEvent.getByToken(etlRecoClustersToken_, etlRecoClusters);

  edm::Handle<MtdSimMergedClusterCollection> simMergedClusters;
  iEvent.getByToken(simMergedClustersToken_, simMergedClusters);

  // associate reco clus to sim merged clus
  reco::MergedRecoToSimCollectionMtd recoToSimColl =
      theAssociator->associateRecoToSim(btlRecoClusters, etlRecoClusters, simMergedClusters);
  reco::MergedSimToRecoCollectionMtd simToRecoColl =
      theAssociator->associateSimToReco(btlRecoClusters, etlRecoClusters, simMergedClusters);

  auto r2s = std::make_unique<reco::MergedRecoToSimCollectionMtd>(recoToSimColl);
  auto s2r = std::make_unique<reco::MergedSimToRecoCollectionMtd>(simToRecoColl);

  iEvent.put(std::move(r2s));
  iEvent.put(std::move(s2r));
}

void MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("associator", edm::InputTag("mtdRecoMergedClusterToSimMergedClusterAssociatorByHits"));
  desc.add<edm::InputTag>("mtdSimMergedClustersTag", edm::InputTag("mtdSimMergedClusterProducer", ""));
  desc.add<edm::InputTag>("btlRecoClustersTag", edm::InputTag("mtdMergedClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("etlRecoClustersTag", edm::InputTag("mtdMergedClusters", "FTLEndcap"));

  cfg.add("mtdRecoMergedClusterToSimMergedClusterAssociationDefault", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MtdRecoMergedClusterToSimMergedClusterAssociatorEDProducer);
