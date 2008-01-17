// -*- C++ -*-
//
//
// Original Author:  Loic QUERTENMONT
//         Created:  Wed Nov  7 17:30:40 CET 2007
// $Id: HSCP_Trigger_MainFunctions.cc,v 1.1 2007/12/13 06:53:14 querten Exp $
//
//

#include "SUSYBSMAnalysis/HSCP/interface/HSCP_Trigger_MainFunctions.h"


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
        if(HSCP_Trigger_L1InterestingPath(i) && TriggerBits[i]) return true;
   }
   return false;
}

bool 
HSCP_Trigger_L1InterestingPath(int Path)
{
    if(Path == l1extra::L1ParticleMap::kSingleMu7)                return true;
    if(Path == l1extra::L1ParticleMap::kSingleIsoEG12)            return true;
    if(Path == l1extra::L1ParticleMap::kSingleEG15)               return true;
    if(Path == l1extra::L1ParticleMap::kSingleJet100)             return true;
    if(Path == l1extra::L1ParticleMap::kSingleTauJet80)           return true;
//    if(Path == l1extra::L1ParticleMap::kHTT200)                   return true;
    if(Path == l1extra::L1ParticleMap::kHTT250)                   return true;
    if(Path == l1extra::L1ParticleMap::kETM30)                    return true;
    if(Path == l1extra::L1ParticleMap::kDoubleMu3)                return true;
    if(Path == l1extra::L1ParticleMap::kDoubleEG10)               return true;
    if(Path == l1extra::L1ParticleMap::kDoubleJet70)              return true;
    if(Path == l1extra::L1ParticleMap::kDoubleTauJet40)           return true;
//    if(Path == l1extra::L1ParticleMap::kMu3_IsoEG5)               return true;
//    if(Path == l1extra::L1ParticleMap::kMu3_EG12)                 return true;
//    if(Path == l1extra::L1ParticleMap::kMu5_Jet15)                return true;
//    if(Path == l1extra::L1ParticleMap::kMu5_TauJet20)             return true;
    if(Path == l1extra::L1ParticleMap::kIsoEG10_Jet20)            return true;
//    if(Path == l1extra::L1ParticleMap::kTripleMu3)                return true;
    if(Path == l1extra::L1ParticleMap::kTripleJet50)              return true;
    if(Path == l1extra::L1ParticleMap::kQuadJet30)                return true;
//    if(Path == l1extra::L1ParticleMap::kExclusiveDoubleIsoEG4)    return true;
//    if(Path == l1extra::L1ParticleMap::kExclusiveDoubleJet60)     return true;
//    if(Path == l1extra::L1ParticleMap::kExclusiveJet25_Gap_Jet25) return true;
//    if(Path == l1extra::L1ParticleMap::kIsoEG10_Jet20_ForJet10)   return true;

    return false;
}



bool
HSCP_Trigger_HLTGlobalDecision(bool* TriggerBits, unsigned int HLT_NPath)
{
   for(unsigned int i=0;i<HLT_NPath;i++){
        if(HSCP_Trigger_HLTInterestingPath(i) && TriggerBits[i]) return true;
   }
   return false;
}


bool
HSCP_Trigger_IsL1ConditionTrue(int Path, bool* TriggerBits)
{
        if(Path == 0  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;        	    //HLT1jet
        if(Path == 1  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150] 
	              && TriggerBits[l1extra::L1ParticleMap::kDoubleJet70])  return true;              //HLT2jet
        if(Path == 2  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT3jet
        if(Path == 2  && TriggerBits[l1extra::L1ParticleMap::kDoubleJet70])  return true;              //HLT3jet
        if(Path == 2  && TriggerBits[l1extra::L1ParticleMap::kTripleJet50])  return true;              //HLT3jet
        if(Path == 3  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT4jet
        if(Path == 3  && TriggerBits[l1extra::L1ParticleMap::kDoubleJet70])  return true;              //HLT4jet
        if(Path == 3  && TriggerBits[l1extra::L1ParticleMap::kTripleJet50])  return true;              //HLT4jet
        if(Path == 3  && TriggerBits[l1extra::L1ParticleMap::kQuadJet30])    return true;              //HLT4jet
        if(Path == 4  && TriggerBits[l1extra::L1ParticleMap::kETM40])        return true;              //HLT1MET
        if(Path == 5  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]
                      && TriggerBits[l1extra::L1ParticleMap::kDoubleJet70])  return true;              //HLT2jetAco
        if(Path == 6  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT1jet1METAco
        if(Path == 7  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT1jet1MET
        if(Path == 8  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT2jet1MET
        if(Path == 9  && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT3jet1MET
        if(Path == 10 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLT4jet1MET
        if(Path == 11 && TriggerBits[l1extra::L1ParticleMap::kHTT300])       return true;              //HLT1MET1HT
        if(Path == 12 && TriggerBits[l1extra::L1ParticleMap::kHTT200])       return true;              //CandHLT1SumET
        if(Path == 13 && TriggerBits[l1extra::L1ParticleMap::kSingleJet100]) return true;              //HLT1jetPE1
        if(Path == 14 && TriggerBits[l1extra::L1ParticleMap::kSingleJet70])  return true;              //HLT1jetPE3
        if(Path == 15 && TriggerBits[l1extra::L1ParticleMap::kSingleJet30])  return true;              //HLT1jetPE5
        if(Path == 16 && TriggerBits[l1extra::L1ParticleMap::kSingleJet15])  return true;              //CandHLT1jetPE7
        if(Path == 17 && TriggerBits[l1extra::L1ParticleMap::kETM20])        return true;              //CandHLT1METPre1
        if(Path == 18 && TriggerBits[l1extra::L1ParticleMap::kMinBias_HTT10])return true;              //CandHLT1METPre2
        if(Path == 19 && TriggerBits[l1extra::L1ParticleMap::kMinBias_HTT10])return true;              //CandHLT1METPre3
        if(Path == 20 && TriggerBits[l1extra::L1ParticleMap::kSingleJet15])  return true;              //CandHLT2jetAve30
        if(Path == 21 && TriggerBits[l1extra::L1ParticleMap::kSingleJet30])  return true;              //CandHLT2jetAve60
        if(Path == 22 && TriggerBits[l1extra::L1ParticleMap::kSingleJet70])  return true;              //CandHLT2jetAve110
        if(Path == 23 && TriggerBits[l1extra::L1ParticleMap::kSingleJet100]) return true;              //CandHLT2jetAve150
        if(Path == 24 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //CandHLT2jetAve200
        if(Path == 25 && TriggerBits[l1extra::L1ParticleMap::kETM30])        return true;              //HLT2jetvbfMET
        if(Path == 26 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTS2jet1METNV
        if(Path == 27 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTS2jet1METAco
        if(Path == 28 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //CandHLTSjet1MET1Aco
        if(Path == 29 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //CandHLTSjet2MET1Aco
        if(Path == 30 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //CandHLTS2jetAco
        if(Path == 31 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_Jet20_ForJet10])return true;     //CandHLTJetMETRapidityGap
        if(Path == 32 && TriggerBits[l1extra::L1ParticleMap::kSingleIsoEG12])return true;              //HLT1Electron
        if(Path == 33 && TriggerBits[l1extra::L1ParticleMap::kSingleEG15])   return true;              //HLT1ElectronRelaxed
        if(Path == 34 && TriggerBits[l1extra::L1ParticleMap::kDoubleIsoEG8]) return true;              //HLT2Electron
        if(Path == 35 && TriggerBits[l1extra::L1ParticleMap::kDoubleEG10])   return true;              //HLT2ElectronRelaxed
        if(Path == 36 && TriggerBits[l1extra::L1ParticleMap::kSingleIsoEG12])return true;              //HLT1Photon
        if(Path == 37 && TriggerBits[l1extra::L1ParticleMap::kSingleEG15])   return true;              //HLT1PhotonRelaxed
        if(Path == 38 && TriggerBits[l1extra::L1ParticleMap::kDoubleIsoEG8]) return true;              //HLT2Photon
        if(Path == 39 && TriggerBits[l1extra::L1ParticleMap::kDoubleEG10])   return true;              //HLT2PhotonRelaxed
        if(Path == 40 && TriggerBits[l1extra::L1ParticleMap::kSingleEG15])   return true;              //HLT1EMHighEt
        if(Path == 41 && TriggerBits[l1extra::L1ParticleMap::kSingleEG15])   return true;              //HLT1EMVeryHighEt
        if(Path == 42 && TriggerBits[l1extra::L1ParticleMap::kDoubleIsoEG8]) return true;              //CandHLT2ElectronZCounter
        if(Path == 43 && TriggerBits[l1extra::L1ParticleMap::kExclusiveDoubleIsoEG6]) return true;     //CandHLT2ElectronExclusive
        if(Path == 44 && TriggerBits[l1extra::L1ParticleMap::kExclusiveDoubleIsoEG6]) return true;     //CandHLT2PhotonExclusive
        if(Path == 45 && TriggerBits[l1extra::L1ParticleMap::kSingleIsoEG10])return true;              //CandHLT1PhotonL1Isolated
        if(Path == 46 && TriggerBits[l1extra::L1ParticleMap::kSingleMu7])    return true;              //HLT1MuonIso
        if(Path == 47 && TriggerBits[l1extra::L1ParticleMap::kSingleMu7])    return true;              //HLT1MuonNonIso
        if(Path == 48 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //CandHLT2MuonIso
        if(Path == 49 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //HLT2MuonNonIso
        if(Path == 50 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //HLT2MuonJPsi
        if(Path == 51 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //HLT2MuonUpsilon
        if(Path == 52 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //HLT2MuonZ
        if(Path == 53 && TriggerBits[l1extra::L1ParticleMap::kTripleMu3])    return true;              //HLTNMuonNonIso
        if(Path == 54 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //HLT2MuonSameSign
        if(Path == 55 && TriggerBits[l1extra::L1ParticleMap::kSingleMu3])    return true;              //CandHLT1MuonPrescalePt3
        if(Path == 56 && TriggerBits[l1extra::L1ParticleMap::kSingleMu5])    return true;              //CandHLT1MuonPrescalePt5
        if(Path == 57 && TriggerBits[l1extra::L1ParticleMap::kSingleMu7])    return true;              //CandHLT1MuonPrescalePt7x7
        if(Path == 58 && TriggerBits[l1extra::L1ParticleMap::kSingleMu7])    return true;              //CandHLT1MuonPrescalePt7x10
        if(Path == 59 && TriggerBits[l1extra::L1ParticleMap::kSingleMu3])    return true;              //CandHLT1MuonLevel1
        if(Path == 59 && TriggerBits[l1extra::L1ParticleMap::kSingleMu5])    return true;              //CandHLT1MuonLevel1
        if(Path == 59 && TriggerBits[l1extra::L1ParticleMap::kSingleMu7])    return true;              //CandHLT1MuonLevel1
        if(Path == 59 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //CandHLT1MuonLevel1
        if(Path == 60 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTB1Jet
        if(Path == 60 && TriggerBits[l1extra::L1ParticleMap::kDoubleJet100]) return true;              //HLTB1Jet
        if(Path == 60 && TriggerBits[l1extra::L1ParticleMap::kTripleJet50] ) return true;              //HLTB1Jet
        if(Path == 60 && TriggerBits[l1extra::L1ParticleMap::kQuadJet30]   ) return true;              //HLTB1Jet
        if(Path == 61 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTB2Jet
        if(Path == 61 && TriggerBits[l1extra::L1ParticleMap::kDoubleJet100]) return true;              //HLTB2Jet
        if(Path == 61 && TriggerBits[l1extra::L1ParticleMap::kTripleJet50] ) return true;              //HLTB2Jet
        if(Path == 61 && TriggerBits[l1extra::L1ParticleMap::kQuadJet30]   ) return true;              //HLTB2Jet
        if(Path == 62 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTB3Jet
        if(Path == 62 && TriggerBits[l1extra::L1ParticleMap::kDoubleJet100]) return true;              //HLTB3Jet
        if(Path == 62 && TriggerBits[l1extra::L1ParticleMap::kTripleJet50] ) return true;              //HLTB3Jet
        if(Path == 62 && TriggerBits[l1extra::L1ParticleMap::kQuadJet30]   ) return true;              //HLTB3Jet
        if(Path == 63 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTB4Jet
        if(Path == 63 && TriggerBits[l1extra::L1ParticleMap::kDoubleJet100]) return true;              //HLTB4Jet
        if(Path == 63 && TriggerBits[l1extra::L1ParticleMap::kTripleJet50] ) return true;              //HLTB4Jet
        if(Path == 63 && TriggerBits[l1extra::L1ParticleMap::kQuadJet30]   ) return true;              //HLTB4Jet
        if(Path == 64 && TriggerBits[l1extra::L1ParticleMap::kSingleJet150]) return true;              //HLTBHT
        if(Path == 64 && TriggerBits[l1extra::L1ParticleMap::kDoubleJet100]) return true;              //HLTBHT
        if(Path == 64 && TriggerBits[l1extra::L1ParticleMap::kTripleJet50] ) return true;              //HLTBHT
        if(Path == 64 && TriggerBits[l1extra::L1ParticleMap::kQuadJet30]   ) return true;              //HLTBHT
        if(Path == 65 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTB1JetMu
        if(Path == 66 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTB2JetMu
        if(Path == 67 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTB3JetMu
        if(Path == 68 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTB4JetMu
        if(Path == 69 && TriggerBits[l1extra::L1ParticleMap::kHTT250])       return true;              //HLTBHTMu
        if(Path == 70 && TriggerBits[l1extra::L1ParticleMap::kDoubleMu3])    return true;              //HLTBJPsiMuMu
        if(Path == 71 && TriggerBits[l1extra::L1ParticleMap::kSingleTauJet80]) return true;            //HLT1Tau
        if(Path == 72 && TriggerBits[l1extra::L1ParticleMap::kTauJet30_ETM30]) return true;            //HLT1Tau1MET
        if(Path == 73 && TriggerBits[l1extra::L1ParticleMap::kDoubleTauJet40]) return true;            //HLT2TauPixel
        if(Path == 74 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_Jet20]) return true;             //HLTXElectronBJet
        if(Path == 75 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTXMuonBJet
        if(Path == 76 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTXMuonBJetSoftMuon
        if(Path == 77 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_Jet30])return true;              //HLTXElectron1Jet
        if(Path == 78 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_Jet15])return true;              //HLTXElectron2Jet
        if(Path == 79 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_Jet15])return true;              //HLTXElectron3Jet
        if(Path == 80 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_Jet15])return true;              //HLTXElectron4Jet
        if(Path == 81 && TriggerBits[l1extra::L1ParticleMap::kMu5_Jet15])    return true;              //HLTXMuonJets
        if(Path == 82 && TriggerBits[l1extra::L1ParticleMap::kMu3_IsoEG5])   return true;              //HLTXElectronMuon
        if(Path == 83 && TriggerBits[l1extra::L1ParticleMap::kMu3_EG12])     return true;              //HLTXElectronMuonRelaxed
        if(Path == 84 && TriggerBits[l1extra::L1ParticleMap::kIsoEG10_TauJet20]) return true;          //HLTXElectronTau
        if(Path == 85 && TriggerBits[l1extra::L1ParticleMap::kMu5_TauJet20]) return true;              //HLTXMuonTau
        if(Path == 86 && TriggerBits[l1extra::L1ParticleMap::kSingleJet100]) return true;              //CandHLTHcalIsolatedTrack
        if(Path == 87 && TriggerBits[l1extra::L1ParticleMap::kMinBias_HTT10])return true;              //HLTMinBiasPixel
        if(Path == 88 && TriggerBits[l1extra::L1ParticleMap::kZeroBias])     return true;              //HLTMinBias
        if(Path == 89 && TriggerBits[l1extra::L1ParticleMap::kZeroBias])     return true;              //HLTZeroBias

    	return false;
}



bool
HSCP_Trigger_HLTInterestingPath(int Path)
{
        if(Path == 0)  return true;              //HLT1jet
        if(Path == 1)  return true;              //HLT2jet
        if(Path == 2)  return true;              //HLT3jet
        if(Path == 3)  return true;              //HLT4jet
        if(Path == 4)  return true;              //HLT1MET
        if(Path == 5)  return true;              //HLT2jetAco
        if(Path == 6)  return true;              //HLT1jet1METAco
        if(Path == 7)  return true;              //HLT1jet1MET
        if(Path == 8)  return true;              //HLT2jet1MET
        if(Path == 9)  return true;              //HLT3jet1MET
        if(Path == 10) return true;              //HLT4jet1MET
        if(Path == 11) return true;              //HLT1MET1HT
        if(Path == 12) return true;              //CandHLT1SumET
//        if(Path == 13) return true;              //HLT1jetPE1
//        if(Path == 14) return true;              //HLT1jetPE3
//        if(Path == 15) return true;              //HLT1jetPE5
//        if(Path == 16) return true;              //CandHLT1jetPE7
//        if(Path == 17) return true;              //CandHLT1METPre1
//        if(Path == 18) return true;              //CandHLT1METPre2
//        if(Path == 19) return true;              //CandHLT1METPre3
//        if(Path == 20) return true;              //CandHLT2jetAve30
//        if(Path == 21) return true;              //CandHLT2jetAve60
//        if(Path == 22) return true;              //CandHLT2jetAve110
//        if(Path == 23) return true;              //CandHLT2jetAve150
        if(Path == 24) return true;              //CandHLT2jetAve200
        if(Path == 25) return true;              //HLT2jetvbfMET
        if(Path == 26) return true;              //HLTS2jet1METNV
        if(Path == 27) return true;              //HLTS2jet1METAco
        if(Path == 28) return true;              //CandHLTSjet1MET1Aco
        if(Path == 29) return true;              //CandHLTSjet2MET1Aco
        if(Path == 30) return true;              //CandHLTS2jetAco
        if(Path == 31) return true;              //CandHLTJetMETRapidityGap
        if(Path == 32) return true;              //HLT1Electron
        if(Path == 33) return true;              //HLT1ElectronRelaxed
        if(Path == 34) return true;              //HLT2Electron
        if(Path == 35) return true;              //HLT2ElectronRelaxed
        if(Path == 36) return true;              //HLT1Photon
        if(Path == 37) return true;              //HLT1PhotonRelaxed
        if(Path == 38) return true;              //HLT2Photon
        if(Path == 39) return true;              //HLT2PhotonRelaxed
        if(Path == 40) return true;              //HLT1EMHighEt
        if(Path == 41) return true;              //HLT1EMVeryHighEt
        if(Path == 42) return true;              //CandHLT2ElectronZCounter
        if(Path == 43) return true;              //CandHLT2ElectronExclusive
        if(Path == 44) return true;              //CandHLT2PhotonExclusive
        if(Path == 45) return true;              //CandHLT1PhotonL1Isolated
        if(Path == 46) return true;              //HLT1MuonIso
        if(Path == 47) return true;              //HLT1MuonNonIso
        if(Path == 48) return true;              //CandHLT2MuonIso
        if(Path == 49) return true;              //HLT2MuonNonIso
        if(Path == 50) return true;              //HLT2MuonJPsi
        if(Path == 51) return true;              //HLT2MuonUpsilon
        if(Path == 52) return true;              //HLT2MuonZ
        if(Path == 53) return true;              //HLTNMuonNonIso
        if(Path == 54) return true;              //HLT2MuonSameSign
//        if(Path == 55) return true;              //CandHLT1MuonPrescalePt3
//        if(Path == 56) return true;              //CandHLT1MuonPrescalePt5
//        if(Path == 57) return true;              //CandHLT1MuonPrescalePt7x7
//        if(Path == 58) return true;              //CandHLT1MuonPrescalePt7x10
//        if(Path == 59) return true;              //CandHLT1MuonLevel1
        if(Path == 60) return true;              //HLTB1Jet
        if(Path == 61) return true;              //HLTB2Jet
        if(Path == 62) return true;              //HLTB3Jet
        if(Path == 63) return true;              //HLTB4Jet
        if(Path == 64) return true;              //HLTBHT
//        if(Path == 65) return true;              //HLTB1JetMu
        if(Path == 66) return true;              //HLTB2JetMu
        if(Path == 67) return true;              //HLTB3JetMu
        if(Path == 68) return true;              //HLTB4JetMu
        if(Path == 69) return true;              //HLTBHTMu
        if(Path == 70) return true;              //HLTBJPsiMuMu
        if(Path == 71) return true;              //HLT1Tau
        if(Path == 72) return true;              //HLT1Tau1MET
        if(Path == 73) return true;              //HLT2TauPixel
        if(Path == 74) return true;              //HLTXElectronBJet
        if(Path == 75) return true;              //HLTXMuonBJet
        if(Path == 76) return true;              //HLTXMuonBJetSoftMuon
        if(Path == 77) return true;              //HLTXElectron1Jet
        if(Path == 78) return true;              //HLTXElectron2Jet
        if(Path == 79) return true;              //HLTXElectron3Jet
        if(Path == 80) return true;              //HLTXElectron4Jet
        if(Path == 81) return true;              //HLTXMuonJets
        if(Path == 82) return true;              //HLTXElectronMuon
        if(Path == 83) return true;              //HLTXElectronMuonRelaxed
        if(Path == 84) return true;              //HLTXElectronTau
        if(Path == 85) return true;              //HLTXMuonTau
        if(Path == 86) return true;              //CandHLTHcalIsolatedTrack
//        if(Path == 87) return true;              //HLTMinBiasPixel
//        if(Path == 88) return true;              //HLTMinBias
//        if(Path == 89) return true;              //HLTZeroBias

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





