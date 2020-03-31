// -*- C++ -*-
//
// Package:    TauValidationMiniAOD
// Class:      TauValidationMiniAOD
//
/**\class TauValidationMiniAOD TauValidationMiniAOD.cc

 Description: <one line class summary>

 Class used to do the Validation of the Tau in miniAOD

 Implementation:
 <Notes on implementation>
 */
//
// Original Author:  Aniello Spiezia
//         Created:  August 13, 2019

#include "Validation/RecoTau/interface/TauValidationMiniAOD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;

TauValidationMiniAOD::TauValidationMiniAOD(const edm::ParameterSet& iConfig) {
  tauCollection_ = consumes<pat::TauCollection>(iConfig.getParameter<InputTag>("tauCollection"));
  refCollectionInputTagToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<InputTag>("RefCollection"));
  extensionName_ = iConfig.getParameter<string>("ExtensionName");
  discriminators_ = iConfig.getParameter<std::vector<edm::ParameterSet> >("discriminators");
}

TauValidationMiniAOD::~TauValidationMiniAOD() {}

void TauValidationMiniAOD::bookHistograms(DQMStore::IBooker& ibooker,
                                          edm::Run const& iRun,
                                          edm::EventSetup const& /* iSetup */) {
  //summary monitor elements
  MonitorElement *ptTight, *etaTight, *phiTight, *massTight, *ptTemp , *etaTemp, *phiTemp, *massTemp, *decayModeFindingTemp, *decayModeTemp,
      *byDeepTau2017v2p1VSerawTemp, *byDeepTau2017v2p1VSjetrawTemp, *byDeepTau2017v2p1VSmurawTemp, *summaryTemp;

  std::cout << "extensionName_: \n";
  std::cout<< extensionName_ ; 
  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/Summary");

  //summary plots
  histoInfo summaryHinfo = (histoSettings_.exists("summary"))
                               ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("summary"))
                               : histoInfo(21, -0.5, 20.5);
  summaryTemp =
      ibooker.book1D("summaryPlotNum", "summaryPlotNum", summaryHinfo.nbins, summaryHinfo.min, summaryHinfo.max);
  summaryMap.insert(std::make_pair("Num", summaryTemp));
  summaryTemp =
      ibooker.book1D("summaryPlotDen", "summaryPlotDen", summaryHinfo.nbins, summaryHinfo.min, summaryHinfo.max);
  summaryMap.insert(std::make_pair("Den", summaryTemp));
  summaryTemp = ibooker.book1D("summaryPlot", "summaryPlot", summaryHinfo.nbins, summaryHinfo.min, summaryHinfo.max);
  summaryMap.insert(std::make_pair("", summaryTemp));

  //other plots
  histoInfo ptHinfo = (histoSettings_.exists("pt")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("pt"))
                                                    : histoInfo(200, 0., 1000.);
  histoInfo etaHinfo = (histoSettings_.exists("eta")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("eta"))
                                                      : histoInfo(200, -3, 3.);
  histoInfo phiHinfo = (histoSettings_.exists("phi")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("phi"))
                                                      : histoInfo(200, -3, 3.);
  histoInfo massHinfo = (histoSettings_.exists("mass"))
                            ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("mass"))
                            : histoInfo(200, 0, 10.);
  histoInfo decayModeFindingHinfo = (histoSettings_.exists("decayModeFinding"))
                                        ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("decayModeFinding"))
                                        : histoInfo(2, -0.5, 1.5);
  histoInfo decayModeHinfo = (histoSettings_.exists("decayMode"))
                                 ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("decayMode"))
                                 : histoInfo(11, -0.5, 10.5);
  histoInfo byDeepTau2017v2p1VSerawHinfo =
      (histoSettings_.exists("byDeepTau2017v2p1VSeraw"))
          ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2p1VSeraw"))
          : histoInfo(200, 0., 1.);
  histoInfo byDeepTau2017v2p1VSjetrawHinfo =
      (histoSettings_.exists("byDeepTau2017v2p1VSjetraw"))
          ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2p1VSjetraw"))
          : histoInfo(200, 0., 1.);
  histoInfo byDeepTau2017v2p1VSmurawHinfo =
      (histoSettings_.exists("byDeepTau2017v2p1VSmuraw"))
          ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2p1VSmuraw"))
          : histoInfo(200, 0., 1.);

  int j = 0;
  for (const auto& it : discriminators_) {
    string DiscriminatorLabel = it.getParameter<string>("discriminator");
    std::cout << "Current discriminator miniaod: \n";
    std::cout << DiscriminatorLabel;
    summaryMap.find("Den")->second->setBinLabel(j + 1, DiscriminatorLabel);
    summaryMap.find("Num")->second->setBinLabel(j + 1, DiscriminatorLabel);
    summaryMap.find("")->second->setBinLabel(j + 1, DiscriminatorLabel);

    j = j + 1;
  }

  ptTemp = ibooker.book1D("tau_pt", "tau_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTemp = ibooker.book1D("tau_eta", "tau_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTemp = ibooker.book1D("tau_phi", "tau_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massTemp = ibooker.book1D("tau_mass", "tau_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  decayModeFindingTemp = ibooker.book1D("tau_decayModeFinding",
                                        "tau_decayModeFinding",
                                        decayModeFindingHinfo.nbins,
                                        decayModeFindingHinfo.min,
                                        decayModeFindingHinfo.max);
  decayModeTemp =
      ibooker.book1D("tau_decayMode", "tau_decayMode", decayModeHinfo.nbins, decayModeHinfo.min, decayModeHinfo.max);
  byDeepTau2017v2p1VSerawTemp = ibooker.book1D("tau_byDeepTau2017v2p1VSeraw",
                                               "tau_byDeepTau2017v2p1VSeraw",
                                               byDeepTau2017v2p1VSerawHinfo.nbins,
                                               byDeepTau2017v2p1VSerawHinfo.min,
                                               byDeepTau2017v2p1VSerawHinfo.max);
  byDeepTau2017v2p1VSjetrawTemp = ibooker.book1D("tau_byDeepTau2017v2p1VSjetraw",
                                                 "tau_byDeepTau2017v2p1VSjetraw",
                                                 byDeepTau2017v2p1VSjetrawHinfo.nbins,
                                                 byDeepTau2017v2p1VSjetrawHinfo.min,
                                                 byDeepTau2017v2p1VSjetrawHinfo.max);
  byDeepTau2017v2p1VSmurawTemp = ibooker.book1D("tau_byDeepTau2017v2p1VSmuraw",
                                                "tau_byDeepTau2017v2p1VSmuraw",
                                                byDeepTau2017v2p1VSmurawHinfo.nbins,
                                                byDeepTau2017v2p1VSmurawHinfo.min,
                                                byDeepTau2017v2p1VSmurawHinfo.max);
  ptMap.insert(std::make_pair("", ptTemp));
  etaMap.insert(std::make_pair("", etaTemp));
  phiMap.insert(std::make_pair("", phiTemp));
  massMap.insert(std::make_pair("", massTemp));
  decayModeFindingMap.insert(std::make_pair("", decayModeFindingTemp));
  decayModeMap.insert(std::make_pair("", decayModeTemp));
  byDeepTau2017v2p1VSerawMap.insert(std::make_pair("", byDeepTau2017v2p1VSerawTemp));
  byDeepTau2017v2p1VSjetrawMap.insert(std::make_pair("", byDeepTau2017v2p1VSjetrawTemp));
  byDeepTau2017v2p1VSmurawMap.insert(std::make_pair("", byDeepTau2017v2p1VSmurawTemp));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/againstJet");
  ptTight = ibooker.book1D("tau_tight_pt", "tau_tight_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTight = ibooker.book1D("tau_tight_eta", "tau_tight_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTight = ibooker.book1D("tau_tight_phi", "tau_tight_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massTight = ibooker.book1D("tau_tight_mass", "tau_tight_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptTightMap.insert(std::make_pair("", ptTight));
  etaTightMap.insert(std::make_pair("", etaTight));
  phiTightMap.insert(std::make_pair("", phiTight));
  massTightMap.insert(std::make_pair("", massTight));
  //ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/againstEle");
  //ptTight = ibooker.book1D("tau_tight_pt", "tau_tight_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  //ptTightMap.insert(std::make_pair("", ptTight));
  //ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/againstMu");
  //ptTight = ibooker.book1D("tau_tight_pt", "tau_tight_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  //ptTightMap.insert(std::make_pair("", ptTight));
}

void TauValidationMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<pat::TauCollection> taus;
  bool isTau = iEvent.getByToken(tauCollection_, taus);
  if (!isTau) {
    edm::LogWarning("TauValidationMiniAOD") << " Tau collection not found while running TauValidationMiniAOD.cc ";
    return;
  }
  typedef edm::View<reco::Candidate> refCandidateCollection;
  Handle<refCandidateCollection> ReferenceCollection;
  bool isRef = iEvent.getByToken(refCollectionInputTagToken_, ReferenceCollection);
  if (!isRef) {
    edm::LogWarning("TauValidationMiniAOD") << " Reference collection not found while running TauValidationMiniAOD.cc ";
    return;
  }
  for (refCandidateCollection::const_iterator RefJet = ReferenceCollection->begin();
       RefJet != ReferenceCollection->end();
       RefJet++) {
    float dRmin = 0.15;
    unsigned matchedTauIndex = -99;
    for (unsigned iTau = 0; iTau < taus->size(); iTau++) {
      pat::TauRef tau(taus, iTau);
      //for (pat::TauCollection::const_iterator tau = taus->begin(); tau != taus->end(); tau++) {
      //pat::TauRef matchedTau(*tau);
      float dR = deltaR(tau->eta(), tau->phi(), RefJet->eta(), RefJet->phi());
      if (dR < dRmin) {
        dRmin = dR;
        matchedTauIndex = iTau;
      }
    }
    if (dRmin < 0.15) {
      pat::TauRef matchedTau(taus, matchedTauIndex);

      ptMap.find("")->second->Fill(matchedTau->pt());
      etaMap.find("")->second->Fill(matchedTau->eta());
      phiMap.find("")->second->Fill(matchedTau->phi());
      massMap.find("")->second->Fill(matchedTau->mass());
      decayModeMap.find("")->second->Fill(matchedTau->decayMode());
      if (matchedTau->isTauIDAvailable("decayModeFinding"))
        decayModeFindingMap.find("")->second->Fill(matchedTau->tauID("decayModeFinding"));
      if (matchedTau->isTauIDAvailable("byDeepTau2017v2p1VSeraw"))
        byDeepTau2017v2p1VSerawMap.find("")->second->Fill(matchedTau->tauID("byDeepTau2017v2p1VSeraw"));
      if (matchedTau->isTauIDAvailable("byDeepTau2017v2p1VSjetraw"))
        byDeepTau2017v2p1VSjetrawMap.find("")->second->Fill(matchedTau->tauID("byDeepTau2017v2p1VSjetraw"));
      if (matchedTau->isTauIDAvailable("byDeepTau2017v2p1VSmuraw"))
        byDeepTau2017v2p1VSmurawMap.find("")->second->Fill(matchedTau->tauID("byDeepTau2017v2p1VSmuraw"));
      int j = 0;
      for (const auto& it : discriminators_) {
        string currentDiscriminator = it.getParameter<string>("discriminator");
        double selectionCut = it.getParameter<double>("selectionCut");
        summaryMap.find("Den")->second->Fill(j);
        if (matchedTau->tauID(currentDiscriminator) >= selectionCut)
          summaryMap.find("Num")->second->Fill(j);
        j = j + 1;
      }
      if (matchedTau->tauID("byTightDeepTau2017v2p1VSjet")>=0.5) {
        ptTightMap.find("")->second->Fill(matchedTau->pt());
        etaTightMap.find("")->second->Fill(matchedTau->eta());
        phiTightMap.find("")->second->Fill(matchedTau->phi());
        massTightMap.find("")->second->Fill(matchedTau->mass());
      }
    }
  }
}
