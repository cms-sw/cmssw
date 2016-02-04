#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorMUON.h"

BetaCalculatorMUON::BetaCalculatorMUON(const edm::ParameterSet& iConfig){
//   m_muontiming_dt       = iConfig.getParameter<InputTag >("muontimingDt"      );
//   m_muontiming_csc      = iConfig.getParameter<InputTag >("muontimingCsc"     );
//   m_muontiming_combined = iConfig.getParameter<InputTag >("muontimingCombined");
}


void BetaCalculatorMUON::addInfoToCandidate(HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   //Do nothing since all muonTiming object are external and get be accessed via reference
   return;
/*
   if(!candidate.hasMuonRef())return;
   reco::MuonRef muon  = candidate.muonRef();

   Handle<reco::MuonTimeExtraMap> timeMap_Dt_h;
   iEvent.getByLabel(m_muontiming_dt,timeMap_Dt_h);
   const reco::MuonTimeExtraMap& timeMap_Dt = *timeMap_Dt_h;
   
   Handle<reco::MuonTimeExtraMap> timeMap_Csc_h;
   iEvent.getByLabel(m_muontiming_dt,timeMap_Csc_h);
   const reco::MuonTimeExtraMap& timeMap_Csc = *timeMap_Csc_h;

   Handle<reco::MuonTimeExtraMap> timeMap_Combined_h;
   iEvent.getByLabel(m_muontiming_dt,timeMap_Combined_h);
   const reco::MuonTimeExtraMap& timeMap_Combined = *timeMap_Combined_h;

   candidate.setMuonTimeDt      (timeMap_Dt      [muon]);
   candidate.setMuonTimeCsc     (timeMap_Csc     [muon]);
   candidate.setMuonTimeCombined(timeMap_Combined[muon]);
*/
}

