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

  MonitorElement *ptTightvsJet, *etaTightvsJet, *phiTightvsJet, *massTightvsJet, *ptTightvsEle, *etaTightvsEle, *phiTightvsEle, *massTightvsEle, *ptTightvsMuo, *etaTightvsMuo, *phiTightvsMuo, *massTightvsMuo, 
*ptMediumvsJet, *etaMediumvsJet, *phiMediumvsJet, *massMediumvsJet, *ptMediumvsEle, *etaMediumvsEle, *phiMediumvsEle, *massMediumvsEle, *ptMediumvsMuo, *etaMediumvsMuo, *phiMediumvsMuo, *massMediumvsMuo,
*ptLoosevsJet, *etaLoosevsJet, *phiLoosevsJet, *massLoosevsJet, *ptLoosevsEle, *etaLoosevsEle, *phiLoosevsEle, *massLoosevsEle, *ptLoosevsMuo, *etaLoosevsMuo, *phiLoosevsMuo, *massLoosevsMuo,*ptTemp , *etaTemp, *phiTemp, *massTemp, *decayModeFindingTemp, *decayModeTemp,
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


  //tight histograms
  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsJet/tight");

  ptTightvsJet = ibooker.book1D("tau_tightvsJet_pt", "tau_tightvsJet_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTightvsJet = ibooker.book1D("tau_tightvsJet_eta", "tau_tightvsJet_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTightvsJet = ibooker.book1D("tau_tightvsJet_phi", "tau_tightvsJet_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massTightvsJet = ibooker.book1D("tau_tightvsJet_mass", "tau_tightvsJet_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptTightvsJetMap.insert(std::make_pair("", ptTightvsJet));
  etaTightvsJetMap.insert(std::make_pair("", etaTightvsJet));
  phiTightvsJetMap.insert(std::make_pair("", phiTightvsJet));
  massTightvsJetMap.insert(std::make_pair("", massTightvsJet));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsEle/tight");

  ptTightvsEle = ibooker.book1D("tau_tightvsEle_pt", "tau_tightvsEle_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTightvsEle = ibooker.book1D("tau_tightvsEle_eta", "tau_tightvsEle_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTightvsEle = ibooker.book1D("tau_tightvsEle_phi", "tau_tightvsEle_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massTightvsEle = ibooker.book1D("tau_tightvsEle_mass", "tau_tightvsEle_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptTightvsEleMap.insert(std::make_pair("", ptTightvsEle));
  etaTightvsEleMap.insert(std::make_pair("", etaTightvsEle));
  phiTightvsEleMap.insert(std::make_pair("", phiTightvsEle));
  massTightvsEleMap.insert(std::make_pair("", massTightvsEle));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsMu/tight");

  ptTightvsMuo = ibooker.book1D("tau_tightvsMuo_pt", "tau_tightvsMuo_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTightvsMuo = ibooker.book1D("tau_tightvsMuo_eta", "tau_tightvsMuo_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTightvsMuo = ibooker.book1D("tau_tightvsMuo_phi", "tau_tightvsMuo_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massTightvsMuo = ibooker.book1D("tau_tightvsMuo_mass", "tau_tightvsMuo_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptTightvsMuoMap.insert(std::make_pair("", ptTightvsMuo));
  etaTightvsMuoMap.insert(std::make_pair("", etaTightvsMuo));
  phiTightvsMuoMap.insert(std::make_pair("", phiTightvsMuo));
  massTightvsMuoMap.insert(std::make_pair("", massTightvsMuo));

  //loose histograms

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsJet/medium");

  ptMediumvsJet = ibooker.book1D("tau_tightvsJet_pt", "tau_tightvsJet_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaMediumvsJet = ibooker.book1D("tau_tightvsJet_eta", "tau_tightvsJet_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiMediumvsJet = ibooker.book1D("tau_tightvsJet_phi", "tau_tightvsJet_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massMediumvsJet = ibooker.book1D("tau_tightvsJet_mass", "tau_tightvsJet_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptMediumvsJetMap.insert(std::make_pair("", ptMediumvsJet));
  etaMediumvsJetMap.insert(std::make_pair("", etaMediumvsJet));
  phiMediumvsJetMap.insert(std::make_pair("", phiMediumvsJet));
  massMediumvsJetMap.insert(std::make_pair("", massMediumvsJet));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsEle/medium");

  ptMediumvsEle = ibooker.book1D("tau_tightvsEle_pt", "tau_tightvsEle_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaMediumvsEle = ibooker.book1D("tau_tightvsEle_eta", "tau_tightvsEle_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiMediumvsEle = ibooker.book1D("tau_tightvsEle_phi", "tau_tightvsEle_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massMediumvsEle = ibooker.book1D("tau_tightvsEle_mass", "tau_tightvsEle_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptMediumvsEleMap.insert(std::make_pair("", ptMediumvsEle));
  etaMediumvsEleMap.insert(std::make_pair("", etaMediumvsEle));
  phiMediumvsEleMap.insert(std::make_pair("", phiMediumvsEle));
  massMediumvsEleMap.insert(std::make_pair("", massMediumvsEle));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsMu/medium");

  ptMediumvsMuo = ibooker.book1D("tau_tightvsMuo_pt", "tau_tightvsMuo_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaMediumvsMuo = ibooker.book1D("tau_tightvsMuo_eta", "tau_tightvsMuo_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiMediumvsMuo = ibooker.book1D("tau_tightvsMuo_phi", "tau_tightvsMuo_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massMediumvsMuo = ibooker.book1D("tau_tightvsMuo_mass", "tau_tightvsMuo_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptMediumvsMuoMap.insert(std::make_pair("", ptMediumvsMuo));
  etaMediumvsMuoMap.insert(std::make_pair("", etaMediumvsMuo));
  phiMediumvsMuoMap.insert(std::make_pair("", phiMediumvsMuo));
  massMediumvsMuoMap.insert(std::make_pair("", massMediumvsMuo));

  //medium histograms

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsJet/loose");

  ptLoosevsJet = ibooker.book1D("tau_tightvsJet_pt", "tau_tightvsJet_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaLoosevsJet = ibooker.book1D("tau_tightvsJet_eta", "tau_tightvsJet_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiLoosevsJet = ibooker.book1D("tau_tightvsJet_phi", "tau_tightvsJet_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massLoosevsJet = ibooker.book1D("tau_tightvsJet_mass", "tau_tightvsJet_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptLoosevsJetMap.insert(std::make_pair("", ptLoosevsJet));
  etaLoosevsJetMap.insert(std::make_pair("", etaLoosevsJet));
  phiLoosevsJetMap.insert(std::make_pair("", phiLoosevsJet));
  massLoosevsJetMap.insert(std::make_pair("", massLoosevsJet));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsEle/loose");

  ptLoosevsEle = ibooker.book1D("tau_tightvsEle_pt", "tau_tightvsEle_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaLoosevsEle = ibooker.book1D("tau_tightvsEle_eta", "tau_tightvsEle_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiLoosevsEle = ibooker.book1D("tau_tightvsEle_phi", "tau_tightvsEle_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massLoosevsEle = ibooker.book1D("tau_tightvsEle_mass", "tau_tightvsEle_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptLoosevsEleMap.insert(std::make_pair("", ptLoosevsEle));
  etaLoosevsEleMap.insert(std::make_pair("", etaLoosevsEle));
  phiLoosevsEleMap.insert(std::make_pair("", phiLoosevsEle));
  massLoosevsEleMap.insert(std::make_pair("", massLoosevsEle));

  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/" + extensionName_ + "/vsMu/loose");

  ptLoosevsMuo = ibooker.book1D("tau_tightvsMuo_pt", "tau_tightvsMuo_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaLoosevsMuo = ibooker.book1D("tau_tightvsMuo_eta", "tau_tightvsMuo_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiLoosevsMuo = ibooker.book1D("tau_tightvsMuo_phi", "tau_tightvsMuo_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massLoosevsMuo = ibooker.book1D("tau_tightvsMuo_mass", "tau_tightvsMuo_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  ptLoosevsMuoMap.insert(std::make_pair("", ptLoosevsMuo));
  etaLoosevsMuoMap.insert(std::make_pair("", etaLoosevsMuo));
  phiLoosevsMuoMap.insert(std::make_pair("", phiLoosevsMuo));
  massLoosevsMuoMap.insert(std::make_pair("", massLoosevsMuo));
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
        ptTightvsJetMap.find("")->second->Fill(matchedTau->pt());
        etaTightvsJetMap.find("")->second->Fill(matchedTau->eta());
        phiTightvsJetMap.find("")->second->Fill(matchedTau->phi());
        massTightvsJetMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byTightDeepTau2017v2p1VSe")>=0.5) {
        ptTightvsEleMap.find("")->second->Fill(matchedTau->pt());
        etaTightvsEleMap.find("")->second->Fill(matchedTau->eta());
        phiTightvsEleMap.find("")->second->Fill(matchedTau->phi());
        massTightvsEleMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byTightDeepTau2017v2p1VSmu")>=0.5) {
        ptTightvsMuoMap.find("")->second->Fill(matchedTau->pt());
        etaTightvsMuoMap.find("")->second->Fill(matchedTau->eta());
        phiTightvsMuoMap.find("")->second->Fill(matchedTau->phi());
        massTightvsMuoMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byMediumDeepTau2017v2p1VSjet")>=0.5) {
        ptMediumvsJetMap.find("")->second->Fill(matchedTau->pt());
        etaMediumvsJetMap.find("")->second->Fill(matchedTau->eta());
        phiMediumvsJetMap.find("")->second->Fill(matchedTau->phi());
        massMediumvsJetMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byMediumDeepTau2017v2p1VSe")>=0.5) {
        ptMediumvsEleMap.find("")->second->Fill(matchedTau->pt());
        etaMediumvsEleMap.find("")->second->Fill(matchedTau->eta());
        phiMediumvsEleMap.find("")->second->Fill(matchedTau->phi());
        massMediumvsEleMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byMediumDeepTau2017v2p1VSmu")>=0.5) {
        ptMediumvsMuoMap.find("")->second->Fill(matchedTau->pt());
        etaMediumvsMuoMap.find("")->second->Fill(matchedTau->eta());
        phiMediumvsMuoMap.find("")->second->Fill(matchedTau->phi());
        massMediumvsMuoMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byLooseDeepTau2017v2p1VSjet")>=0.5) {
        ptLoosevsJetMap.find("")->second->Fill(matchedTau->pt());
        etaLoosevsJetMap.find("")->second->Fill(matchedTau->eta());
        phiLoosevsJetMap.find("")->second->Fill(matchedTau->phi());
        massLoosevsJetMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byLooseDeepTau2017v2p1VSe")>=0.5) {
        ptLoosevsEleMap.find("")->second->Fill(matchedTau->pt());
        etaLoosevsEleMap.find("")->second->Fill(matchedTau->eta());
        phiLoosevsEleMap.find("")->second->Fill(matchedTau->phi());
        massLoosevsEleMap.find("")->second->Fill(matchedTau->mass());
      }
      if (matchedTau->tauID("byLooseDeepTau2017v2p1VSmu")>=0.5) {
        ptLoosevsMuoMap.find("")->second->Fill(matchedTau->pt());
        etaLoosevsMuoMap.find("")->second->Fill(matchedTau->eta());
        phiLoosevsMuoMap.find("")->second->Fill(matchedTau->phi());
        massLoosevsMuoMap.find("")->second->Fill(matchedTau->mass());
      }
    }
  }
}
