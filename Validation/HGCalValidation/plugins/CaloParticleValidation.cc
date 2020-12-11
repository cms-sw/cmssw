// -*- C++ -*-
//
// Class:      CaloParticleValidation
// Original Author:  Marco Rovere
// Created:  Thu, 18 Jan 2018 15:54:55 GMT
//
//

#include <string>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//
// class declaration
//

struct Histogram_CaloParticleSingle {
  dqm::reco::MonitorElement* simPFSuperClusterSize_;
  dqm::reco::MonitorElement* simPFSuperClusterEnergy_;
  dqm::reco::MonitorElement* pfcandidateType_;
  dqm::reco::MonitorElement* pfcandidateEnergy_;
  dqm::reco::MonitorElement* pfcandidatePt_;
  dqm::reco::MonitorElement* pfcandidateEta_;
  dqm::reco::MonitorElement* pfcandidatePhi_;
  dqm::reco::MonitorElement* pfcandidateElementsInBlocks_;
  dqm::reco::MonitorElement* pfcandidate_vect_sum_pt_;  // This is indeed a cumulative istogram
};

using Histograms_CaloParticleValidation = std::unordered_map<int, Histogram_CaloParticleSingle>;

class CaloParticleValidation : public DQMGlobalEDAnalyzer<Histograms_CaloParticleValidation> {
public:
  explicit CaloParticleValidation(const edm::ParameterSet&);
  ~CaloParticleValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms_CaloParticleValidation&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms_CaloParticleValidation const&) const override;

  // ----------member data ---------------------------
  std::string folder_;
  edm::EDGetTokenT<std::vector<reco::SuperCluster>> simPFClusters_;
  edm::EDGetTokenT<reco::PFCandidateCollection> simPFCandidates_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloParticleValidation::CaloParticleValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      simPFClusters_(consumes<std::vector<reco::SuperCluster>>(iConfig.getParameter<edm::InputTag>("simPFClusters"))),
      simPFCandidates_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("simPFCandidates"))) {
  //now do what ever initialization is needed
}

CaloParticleValidation::~CaloParticleValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------

void CaloParticleValidation::dqmAnalyze(edm::Event const& iEvent,
                                        edm::EventSetup const& iSetup,
                                        Histograms_CaloParticleValidation const& histos) const {
  using namespace edm;

  Handle<std::vector<reco::SuperCluster>> simPFClustersHandle;
  iEvent.getByToken(simPFClusters_, simPFClustersHandle);
  std::vector<reco::SuperCluster> const& simPFClusters = *simPFClustersHandle;

  Handle<reco::PFCandidateCollection> simPFCandidatesHandle;
  iEvent.getByToken(simPFCandidates_, simPFCandidatesHandle);
  reco::PFCandidateCollection const& simPFCandidates = *simPFCandidatesHandle;

  // simPFSuperClusters
  for (auto const& sc : simPFClusters) {
    histos.at(0).simPFSuperClusterSize_->Fill((float)sc.clustersSize());
    histos.at(0).simPFSuperClusterEnergy_->Fill(sc.rawEnergy());
  }

  // simPFCandidates
  int offset = 100000;
  double ptx_tot = 0.;
  double pty_tot = 0.;
  for (auto const& pfc : simPFCandidates) {
    size_t type = offset + pfc.particleId();
    ptx_tot += pfc.px();
    pty_tot += pfc.py();
    histos.at(offset).pfcandidateType_->Fill(type - offset);
    auto& histo = histos.at(type);
    histo.pfcandidateEnergy_->Fill(pfc.energy());
    histo.pfcandidatePt_->Fill(pfc.pt());
    histo.pfcandidateEta_->Fill(pfc.eta());
    histo.pfcandidatePhi_->Fill(pfc.phi());
    histo.pfcandidateElementsInBlocks_->Fill(pfc.elementsInBlocks().size());
  }
  auto& histo = histos.at(offset);
  histo.pfcandidate_vect_sum_pt_->Fill(std::sqrt(ptx_tot * ptx_tot + pty_tot * pty_tot));
}

void CaloParticleValidation::bookHistograms(DQMStore::IBooker& ibook,
                                            edm::Run const& run,
                                            edm::EventSetup const& iSetup,
                                            Histograms_CaloParticleValidation& histos) const {
  int offset = 100000;
  ibook.setCurrentFolder(folder_ + "PFCandidates");
  histos[offset].pfcandidateType_ = ibook.book1D("PFCandidateType", "PFCandidateType", 10, 0, 10);
  histos[offset].pfcandidate_vect_sum_pt_ = ibook.book1D("PFCandidatePtVectSum", "PFCandidatePtVectSum", 200, 0., 200.);
  for (size_t type = reco::PFCandidate::h; type <= reco::PFCandidate::egamma_HF; type++) {
    ibook.setCurrentFolder(folder_ + "PFCandidates/" + std::to_string(type));
    auto& histo = histos[offset + type];
    histo.pfcandidateEnergy_ = ibook.book1D("PFCandidateEnergy", "PFCandidateEnergy", 250, 0., 250.);
    histo.pfcandidatePt_ = ibook.book1D("PFCandidatePt", "PFCandidatePt", 250, 0., 250.);
    histo.pfcandidateEta_ = ibook.book1D("PFCandidateEta", "PFCandidateEta", 100, -5., 5.);
    histo.pfcandidatePhi_ = ibook.book1D("PFCandidatePhi", "PFCandidatePhi", 100, -4., 4.);
    histo.pfcandidateElementsInBlocks_ = ibook.book1D("PFCandidateElements", "PFCandidateElements", 20, 0., 20.);
  }
  // Folder '0' is meant to be cumulative, with no connection to pdgId
  ibook.setCurrentFolder(folder_ + std::to_string(0));
  histos[0].simPFSuperClusterSize_ = ibook.book1D("SimPFSuperClusterSize", "SimPFSuperClusterSize", 40, 0., 40.);
  histos[0].simPFSuperClusterEnergy_ =
      ibook.book1D("SimPFSuperClusterEnergy", "SimPFSuperClusterEnergy", 250, 0., 500.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CaloParticleValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "HGCAL/");  // Please keep the trailing '/'
  desc.add<edm::InputTag>("simPFClusters", edm::InputTag("simPFProducer", "perfect"));
  desc.add<edm::InputTag>("simPFCandidates", edm::InputTag("simPFProducer"));
  descriptions.add("caloparticlevalidationDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloParticleValidation);
