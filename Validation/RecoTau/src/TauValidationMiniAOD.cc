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
  tauCollection_ = consumes<pat::TauCollection>(iConfig.getParameter<edm::InputTag>("tauCollection"));
}

TauValidationMiniAOD::~TauValidationMiniAOD() {
}

void TauValidationMiniAOD::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & /* iSetup */)
{
  MonitorElement * ptTemp, *etaTemp, *byDeepTau2017v2VSerawTemp, *byDeepTau2017v2VSjetrawTemp, *byDeepTau2017v2VSmurawTemp;
  
  ibooker.setCurrentFolder("RecoTauV/miniAODValidation/");
  
  //Histograms settings
  histoInfo ptHinfo = (histoSettings_.exists("pt")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("pt")) : histoInfo(500, 0., 1000.);
  histoInfo etaHinfo = (histoSettings_.exists("eta")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("eta")) : histoInfo(500, -3, 3.);
  histoInfo byDeepTau2017v2VSerawHinfo = (histoSettings_.exists("byDeepTau2017v2VSeraw")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2VSeraw")) : histoInfo(500, 0., 1.);
  histoInfo byDeepTau2017v2VSjetrawHinfo = (histoSettings_.exists("byDeepTau2017v2VSjetraw")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2VSjetraw")) : histoInfo(500, 0., 1.);
  histoInfo byDeepTau2017v2VSmurawHinfo = (histoSettings_.exists("byDeepTau2017v2VSmuraw")) ? histoInfo(histoSettings_.getParameter<edm::ParameterSet>("byDeepTau2017v2VSmuraw")) : histoInfo(500, 0., 1.);
  ptTemp = ibooker.book1D("tau_pt", "tau_pt", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTemp = ibooker.book1D("tau_eta", "tau_eta", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  byDeepTau2017v2VSerawTemp = ibooker.book1D("tau_byDeepTau2017v2VSeraw", "tau_byDeepTau2017v2VSeraw", byDeepTau2017v2VSerawHinfo.nbins, byDeepTau2017v2VSerawHinfo.min, byDeepTau2017v2VSerawHinfo.max);
  byDeepTau2017v2VSjetrawTemp = ibooker.book1D("tau_byDeepTau2017v2VSjetraw", "tau_byDeepTau2017v2VSjetraw", byDeepTau2017v2VSjetrawHinfo.nbins, byDeepTau2017v2VSjetrawHinfo.min, byDeepTau2017v2VSjetrawHinfo.max);
  byDeepTau2017v2VSmurawTemp = ibooker.book1D("tau_byDeepTau2017v2VSmuraw", "tau_byDeepTau2017v2VSmuraw", byDeepTau2017v2VSmurawHinfo.nbins, byDeepTau2017v2VSmurawHinfo.min, byDeepTau2017v2VSmurawHinfo.max);
  ptTauVisibleMap.insert(std::make_pair("", ptTemp));
  etaTauVisibleMap.insert(std::make_pair("", etaTemp));
  byDeepTau2017v2VSerawVisibleMap.insert(std::make_pair("", byDeepTau2017v2VSerawTemp));
  byDeepTau2017v2VSjetrawVisibleMap.insert(std::make_pair("", byDeepTau2017v2VSjetrawTemp));
  byDeepTau2017v2VSmurawVisibleMap.insert(std::make_pair("", byDeepTau2017v2VSmurawTemp));
}

void TauValidationMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<pat::TauCollection> taus; 
  iEvent.getByToken(tauCollection_, taus);
  for (pat::TauCollection::const_iterator tau = taus->begin(); tau != taus->end(); tau++) {
    ptTauVisibleMap.find("")->second->Fill(tau->pt());  
    etaTauVisibleMap.find("")->second->Fill(tau->eta());  
    if(tau->isTauIDAvailable("byDeepTau2017v2VSeraw"))   byDeepTau2017v2VSerawVisibleMap.find("")->second->Fill(tau->tauID("byDeepTau2017v2VSeraw"));  
    if(tau->isTauIDAvailable("byDeepTau2017v2VSjetraw")) byDeepTau2017v2VSjetrawVisibleMap.find("")->second->Fill(tau->tauID("byDeepTau2017v2VSjetraw"));
    if(tau->isTauIDAvailable("byDeepTau2017v2VSmuraw"))  byDeepTau2017v2VSmurawVisibleMap.find("")->second->Fill(tau->tauID("byDeepTau2017v2VSmuraw"));
  }
}
