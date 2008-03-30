// -*- H -*-
//
//
// Original Author:  Loic QUERTENMONT
//         Created:  Wed Nov  7 17:30:40 CET 2007
// $Id: HSCP_Trigger_MainFunctions.h,v 1.1 2007/12/13 06:53:12 querten Exp $
//
//

#ifndef SUSYBSMANALYSIS_HSCPTRIGGER_MAINFUNCTIONS_H
#define SUSYBSMANALYSIS_HSCPTRIGGER_MAINFUNCTIONS_H


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"


//#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

using namespace edm;


double HSCP_Trigger_DeltaR(  double phi1, double eta1, double phi2, double eta2);
double HSCP_Trigger_DeltaPhi(double phi1,  double phi2);

bool   HSCP_Trigger_L1MuonAbovePtThreshold     (const l1extra::L1MuonParticleCollection L1_Muons,double PtThreshold);
bool   HSCP_Trigger_L1MuonAbovePtThreshold     (const l1extra::L1MuonParticleCollection L1_Muons,double PtThreshold,int* recoL1Muon, double* MinDt, double DeltaTMax);
bool   HSCP_Trigger_L1TwoMuonAbovePtThreshold  (const l1extra::L1MuonParticleCollection L1_Muons,double PtThreshold);
bool   HSCP_Trigger_L1TwoMuonAbovePtThreshold  (const l1extra::L1MuonParticleCollection L1_Muons,double PtThreshold,int* recoL1Muon, double* MinDt, double DeltaTMax);
bool   HSCP_Trigger_L1METAbovePtThreshold      (const l1extra::L1EtMissParticle L1_MET          ,double PtThreshold);
bool   HSCP_Trigger_L1HTTAbovePtThreshold      (const l1extra::L1EtMissParticle L1_MET          ,double PtThreshold);
bool   HSCP_Trigger_L1JetAbovePtThreshold      (const l1extra::L1JetParticleCollection L1_Jets  ,double PtThreshold);

bool   HSCP_Trigger_HLTMuonAbovePtThreshold    (const reco::RecoChargedCandidateCollection HLT_Muons,double PtThreshold);
bool   HSCP_Trigger_HLTMETAbovePtThreshold     (const reco::CaloMETCollection HLT_MET           ,double PtThreshold);
bool   HSCP_Trigger_HLTSumEtAbovePtThreshold   (const reco::CaloMETCollection HLT_MET           ,double PtThreshold);
bool   HSCP_Trigger_HLTJetAbovePtThreshold     (const reco::CaloJetCollection HLT_Jets          ,double PtThreshold);

bool   HSCP_Trigger_L1GlobalDecision           (bool* TriggerBits);
bool   HSCP_Trigger_HLTGlobalDecision          (bool* TriggerBits, unsigned int HLT_NPath);

int    HSCP_Trigger_ClosestHSCP                (double phi, double eta, double dRMax, const reco::CandidateCollection MC_Cand);
int    HSCP_Trigger_ClosestL1Muon              (double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons);
int    HSCP_Trigger_ClosestHLTMuon             (double phi, double eta, double dRMax, const reco::RecoChargedCandidateCollection HLT_Muons);


#endif



