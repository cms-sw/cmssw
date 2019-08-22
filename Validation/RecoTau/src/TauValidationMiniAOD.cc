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

using namespace edm;
using namespace std;
using namespace reco;

TauValidationMiniAOD::TauValidationMiniAOD(const edm::ParameterSet& iConfig)
{
  tauCollection_              = consumes<pat::TauCollection>         (iConfig.getParameter<InputTag>("tauCollection"));
  refCollectionInputTagToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<InputTag>("RefCollection"));
  extensionName_ = iConfig.getParameter<string>("ExtensionName");
}

TauValidationMiniAOD::~TauValidationMiniAOD() {
}

void TauValidationMiniAOD::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & /* iSetup */)
{
  MonitorElement *ptTemp, *etaTemp, *phiTemp, *massTemp, *decayModeFindingTemp, *byDeepTau2017v2VSerawTemp, *byDeepTau2017v2VSjetrawTemp, *byDeepTau2017v2VSmurawTemp;
  
  ibooker.setCurrentFolder("RecoTauV/miniAODValidation"+extensionName_);
  
  //Histograms settings
  histoInfo ptHinfo = (histoSettings_.exists("pt")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("pt")) : histoInfo(200, 0., 1000.);
  histoInfo etaHinfo = (histoSettings_.exists("eta")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("eta")) : histoInfo(200, -3, 3.);
  histoInfo phiHinfo = (histoSettings_.exists("phi")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("phi")) : histoInfo(200, -3, 3.);
  histoInfo massHinfo = (histoSettings_.exists("mass")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("mass")) : histoInfo(200, 0, 10.);
  histoInfo decayModeFindingHinfo = (histoSettings_.exists("decayModeFinding")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("decayModeFinding")) : histoInfo(2, -0.5, 1.5);
  histoInfo byDeepTau2017v2VSerawHinfo = (histoSettings_.exists("byDeepTau2017v2VSeraw")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2VSeraw")) : histoInfo(200, 0., 1.);
  histoInfo byDeepTau2017v2VSjetrawHinfo = (histoSettings_.exists("byDeepTau2017v2VSjetraw")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2VSjetraw")) : histoInfo(200, 0., 1.);
  histoInfo byDeepTau2017v2VSmurawHinfo = (histoSettings_.exists("byDeepTau2017v2VSmuraw")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2VSmuraw")) : histoInfo(200, 0., 1.);
  ptTemp = ibooker.book1D("tau_pt", "tau_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTemp = ibooker.book1D("tau_eta", "tau_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTemp = ibooker.book1D("tau_phi", "tau_phi", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  massTemp = ibooker.book1D("tau_mass", "tau_mass", massHinfo.nbins, massHinfo.min, massHinfo.max);
  decayModeFindingTemp = ibooker.book1D("tau_decayModeFinding", "tau_decayModeFinding", decayModeFindingHinfo.nbins, decayModeFindingHinfo.min, decayModeFindingHinfo.max);
  byDeepTau2017v2VSerawTemp = ibooker.book1D("tau_byDeepTau2017v2VSeraw", "tau_byDeepTau2017v2VSeraw", byDeepTau2017v2VSerawHinfo.nbins, byDeepTau2017v2VSerawHinfo.min, byDeepTau2017v2VSerawHinfo.max);
  byDeepTau2017v2VSjetrawTemp = ibooker.book1D("tau_byDeepTau2017v2VSjetraw", "tau_byDeepTau2017v2VSjetraw", byDeepTau2017v2VSjetrawHinfo.nbins, byDeepTau2017v2VSjetrawHinfo.min, byDeepTau2017v2VSjetrawHinfo.max);
  byDeepTau2017v2VSmurawTemp = ibooker.book1D("tau_byDeepTau2017v2VSmuraw", "tau_byDeepTau2017v2VSmuraw", byDeepTau2017v2VSmurawHinfo.nbins, byDeepTau2017v2VSmurawHinfo.min, byDeepTau2017v2VSmurawHinfo.max);
  ptTauVisibleMap.insert(std::make_pair("", ptTemp));
  etaTauVisibleMap.insert(std::make_pair("", etaTemp));
  phiTauVisibleMap.insert(std::make_pair("", phiTemp));
  massTauVisibleMap.insert(std::make_pair("", massTemp));
  decayModeFindingTauVisibleMap.insert(std::make_pair("", decayModeFindingTemp));
  byDeepTau2017v2VSerawVisibleMap.insert(std::make_pair("", byDeepTau2017v2VSerawTemp));
  byDeepTau2017v2VSjetrawVisibleMap.insert(std::make_pair("", byDeepTau2017v2VSjetrawTemp));
  byDeepTau2017v2VSmurawVisibleMap.insert(std::make_pair("", byDeepTau2017v2VSmurawTemp));
}

void TauValidationMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<pat::TauCollection> taus; 
  iEvent.getByToken(tauCollection_, taus);
  typedef edm::View<reco::Candidate> refCandidateCollection;
  Handle<refCandidateCollection> ReferenceCollection;
  bool isRef = iEvent.getByToken( refCollectionInputTagToken_, ReferenceCollection );
  if (!isRef) {
    std::cerr << " Reference collection  not found while running TauTagValidation.cc " << std::endl;
    return;
  }
  for (refCandidateCollection::const_iterator RefJet= ReferenceCollection->begin() ; RefJet != ReferenceCollection->end(); RefJet++ ){
    float dRmin = 0.15;
    float taupt = -99.;
    float taueta = -99.;
    float tauphi = -99.;
    float taumass = -99.;
    float taudecay = -99.;
    float taudiscr1 = -99.;
    float taudiscr2 = -99.;
    float taudiscr3 = -99.;
    for (pat::TauCollection::const_iterator tau = taus->begin(); tau != taus->end(); tau++) {
      float dR = deltaR(tau->eta(), tau->phi(), RefJet->eta(), RefJet->phi());
      if(dR<dRmin){
	dRmin = dR;
	taupt = tau->pt();
	taueta = tau->eta();
	tauphi = tau->phi();
	taumass = tau->mass();
	if(tau->isTauIDAvailable("decayModeFinding"))        taudecay  = tau->tauID("decayModeFinding");
	if(tau->isTauIDAvailable("byDeepTau2017v2VSeraw"))   taudiscr1 = tau->tauID("byDeepTau2017v2VSeraw");
	if(tau->isTauIDAvailable("byDeepTau2017v2VSjetraw")) taudiscr2 = tau->tauID("byDeepTau2017v2VSjetraw");
	if(tau->isTauIDAvailable("byDeepTau2017v2VSmuraw"))  taudiscr3 = tau->tauID("byDeepTau2017v2VSmuraw");
      }
    }
    if(taupt>0){
      ptTauVisibleMap.find("")->second->Fill(taupt);  
      etaTauVisibleMap.find("")->second->Fill(taueta);  
      phiTauVisibleMap.find("")->second->Fill(tauphi);  
      massTauVisibleMap.find("")->second->Fill(taumass);  
      decayModeFindingTauVisibleMap.find("")->second->Fill(taudecay);  
      byDeepTau2017v2VSerawVisibleMap.find("")->second->Fill(taudiscr1);  
      byDeepTau2017v2VSjetrawVisibleMap.find("")->second->Fill(taudiscr2);
      byDeepTau2017v2VSmurawVisibleMap.find("")->second->Fill(taudiscr3);
    }
  }
}
