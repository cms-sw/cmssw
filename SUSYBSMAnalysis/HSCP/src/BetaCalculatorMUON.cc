#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorMUON.h"

Beta_Calculator_MUON::Beta_Calculator_MUON(const edm::ParameterSet& iConfig){
  m_muonsTOFTag = iConfig.getParameter<edm::InputTag>("muontiming");
}


void Beta_Calculator_MUON::addInfoToCandidate(HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if(!candidate.isMuon())return;

   Handle<reco::MuonTimeExtraMap> timeMap_h;
   iEvent.getByLabel(m_muonsTOFTag,timeMap_h);
   const reco::MuonTimeExtraMap & timeMap = *timeMap_h;

   reco::MuonRef muon  = candidate.getMuon();
   
   DriftTubeTOF result;
   result.invBeta        = timeMap[muon].inverseBeta();
   result.invBetaErr     = timeMap[muon].inverseBetaErr();
   result.invBetaFree    = timeMap[muon].freeInverseBeta();
   result.invBetaFreeErr = timeMap[muon].freeInverseBetaErr(); 

   candidate.setDt(result);
}

