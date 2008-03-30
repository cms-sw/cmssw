// -*- C++ -*-
//
//
// Original Author:  Loic QUERTENMONT
//         Created:  Wed Nov  7 17:30:40 CET 2007
// $Id: HSCP_Trigger_MainFunctions.cc,v 1.2 2008/01/17 05:12:12 querten Exp $
//
//

#include "SUSYBSMAnalysis/HSCP/interface/Trigger_MainFunctions.h"


using namespace edm;



bool
HSCP_Trigger_L1MuonAbovePtThreshold(const l1extra::L1MuonParticleCollection L1_Muons, double PtThreshold)
{
  for(unsigned int i=0;i<L1_Muons.size();i++){
        if(L1_Muons[i].gmtMuonCand().quality()>=4 && L1_Muons[i].et() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger_L1MuonAbovePtThreshold(const l1extra::L1MuonParticleCollection L1_Muons, double PtThreshold, int* recoL1Muon, double* MinDt,  double DeltaTMax)
{
  for(unsigned int i=0;i<L1_Muons.size();i++){
        if(recoL1Muon[0]==(int)i && MinDt[0] >= DeltaTMax) continue;
        if(recoL1Muon[1]==(int)i && MinDt[1] >= DeltaTMax) continue;
        if(L1_Muons[i].gmtMuonCand().quality()>=4 && L1_Muons[i].et() >= PtThreshold)return true;}
  return false;
}


bool
HSCP_Trigger_L1TwoMuonAbovePtThreshold(const l1extra::L1MuonParticleCollection L1_Muons, double PtThreshold)
{
  int count = 0;
  for(unsigned int i=0;i<L1_Muons.size();i++){
        if(L1_Muons[i].gmtMuonCand().quality()>=3 && L1_Muons[i].gmtMuonCand().quality()!=4 && L1_Muons[i].et() >= PtThreshold)count++;
  }
  if(count >= 2)return true;
  return false;
}



bool
HSCP_Trigger_L1TwoMuonAbovePtThreshold(const l1extra::L1MuonParticleCollection L1_Muons, double PtThreshold, int* recoL1Muon, double* MinDt,  double DeltaTMax)
{
  int count = 0;
  for(unsigned int i=0;i<L1_Muons.size();i++){
        if(recoL1Muon[0]==(int)i && MinDt[0] >= DeltaTMax) continue;
        if(recoL1Muon[1]==(int)i && MinDt[1] >= DeltaTMax) continue;
        if(L1_Muons[i].gmtMuonCand().quality()>=3 && L1_Muons[i].gmtMuonCand().quality()!=4 && L1_Muons[i].et() >= PtThreshold)count++;
  }
  if(count >= 2)return true;
  return false;
}




bool
HSCP_Trigger_L1METAbovePtThreshold(const l1extra::L1EtMissParticle L1_MET, double PtThreshold)
{
  if(L1_MET.etMiss() >= PtThreshold)return true;
  return false;
}

bool
HSCP_Trigger_L1HTTAbovePtThreshold(const l1extra::L1EtMissParticle L1_MET, double PtThreshold)
{
  if(L1_MET.etHad() >= PtThreshold)return true;
  return false;
}


bool
HSCP_Trigger_L1JetAbovePtThreshold(const l1extra::L1JetParticleCollection L1_Jets, double PtThreshold)
{
  for(unsigned int i=0;i<L1_Jets.size();i++){
        if(L1_Jets[i].et() >= PtThreshold)return true; }  
  return false;
}

bool
HSCP_Trigger_HLTMuonAbovePtThreshold(const  reco::RecoChargedCandidateCollection HLT_Muons, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_Muons.size();i++){
        if(HLT_Muons[i].et() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger_HLTMETAbovePtThreshold(const reco::CaloMETCollection HLT_MET, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_MET.size();i++){
	  if(HLT_MET[i].et() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger_HLTSumEtAbovePtThreshold(const reco::CaloMETCollection HLT_MET, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_MET.size();i++){
          if(HLT_MET[i].sumEt() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger_HLTJetAbovePtThreshold(const reco::CaloJetCollection HLT_Jets, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_Jets.size();i++){
        if(HLT_Jets[i].et() >= PtThreshold)return true; }
  return false;
}



bool
HSCP_Trigger_L1GlobalDecision(bool* TriggerBits)
{
   for(unsigned int i=0;i<l1extra::L1ParticleMap::kNumOfL1TriggerTypes;i++){
        if(TriggerBits[i]) return true;
   }
   return false;
}

bool
HSCP_Trigger_HLTGlobalDecision(bool* TriggerBits, unsigned int HLT_NPath)
{
   for(unsigned int i=0;i<HLT_NPath;i++){
        if(TriggerBits[i]) return true;
   }
   return false;
}


int
HSCP_Trigger_ClosestHSCP(double phi, double eta, double dRMax, const reco::CandidateCollection MC_Cand)
{
       double dR = 99999; int J=-1;
       for(unsigned int j=0;j<MC_Cand.size();j++){
          if(dR > HSCP_Trigger_DeltaR(phi,eta, MC_Cand[j].phi(), MC_Cand[j].eta()) ){
             dR = HSCP_Trigger_DeltaR(phi,eta, MC_Cand[j].phi(), MC_Cand[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;	
}

int
HSCP_Trigger_ClosestL1Muon(double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons)
{
       double dR = 99999; int J=-1;
       for(unsigned int j=0;j<L1_Muons.size();j++){
          if(dR > HSCP_Trigger_DeltaR(phi,eta, L1_Muons[j].phi(), L1_Muons[j].eta()) ){
             dR = HSCP_Trigger_DeltaR(phi,eta, L1_Muons[j].phi(), L1_Muons[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;
}

int
HSCP_Trigger_ClosestHLTMuon(double phi, double eta, double dRMax, const  reco::RecoChargedCandidateCollection HLT_Muons)
{
       double dR = 99999; int J=-1;
       for(unsigned int j=0;j<HLT_Muons.size();j++){
          if(dR > HSCP_Trigger_DeltaR(phi,eta, HLT_Muons[j].phi(), HLT_Muons[j].eta()) ){
             dR = HSCP_Trigger_DeltaR(phi,eta, HLT_Muons[j].phi(), HLT_Muons[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;
}


double
HSCP_Trigger_DeltaR(double phi1, double eta1, double phi2, double eta2)
{
        double deltaphi=phi1-phi2;

        if(fabs(deltaphi)>3.14)deltaphi=2*3.14-fabs(deltaphi);
        else if(fabs(deltaphi)<3.14)deltaphi=fabs(deltaphi);
        return (sqrt(pow(deltaphi,2)+pow(eta1 - eta2,2)));
}

double
HSCP_Trigger_DeltaPhi(double phi1,  double phi2)
{
        double deltaphi=phi1-phi2;

        if(fabs(deltaphi)>3.14)deltaphi=2*3.14-fabs(deltaphi);
        else if(fabs(deltaphi)<3.14)deltaphi=fabs(deltaphi);
        return deltaphi;
}





