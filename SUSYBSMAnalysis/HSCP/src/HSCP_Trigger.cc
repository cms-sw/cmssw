// -*- C++ -*-
//
// Package:    HSCP_Trigger
// Class:      HSCP_Trigger
// 
/**\class HSCP_Trigger HSCP_Trigger.cc SUSYBSMAnalysis/HSCP/src/HSCP_Trigger.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic QUERTENMONT
//         Created:  Wed Nov  7 17:30:40 CET 2007
// $Id$
//
//


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


#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
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

#include "SUSYBSMAnalysis/HSCP/interface/SlowHSCPFilter_MainFunctions.h"


#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

using namespace edm;


double DeltaR(  double phi1, double eta1, double phi2, double eta2);
double DeltaPhi(double phi1,  double phi2);
void   DivideAndComputeError(TH1D* Distrib, TH1D* N);
void   ScaleAndComputeError(TH1D* Distrib, unsigned int N);
char*  LatexFormat(const char* txt);


class PtSorter {
 public:
  template <class T> bool operator() ( const T& a, const T& b ) {
     return ( a.pt() > b.pt() );
  }
};



//
// class decleration
//

class HSCP_Trigger : public edm::EDAnalyzer {
   public:
      explicit HSCP_Trigger(const edm::ParameterSet&);
      ~HSCP_Trigger();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      bool L1MuonAbovePtThreshold  (const l1extra::L1MuonParticleCollection L1_Muons,double PtThreshold);
      bool L1TwoMuonAbovePtThreshold  (const l1extra::L1MuonParticleCollection L1_Muons,double PtThreshold);
      bool L1METAbovePtThreshold   (const l1extra::L1EtMissParticle L1_MET          ,double PtThreshold);
      bool L1HTTAbovePtThreshold   (const l1extra::L1EtMissParticle L1_MET          ,double PtThreshold);
      bool L1JetAbovePtThreshold   (const l1extra::L1JetParticleCollection L1_Jets  ,double PtThreshold);

      bool HLTMuonAbovePtThreshold (const reco::RecoChargedCandidateCollection HLT_Muons,double PtThreshold);
      bool HLTMETAbovePtThreshold  (const reco::CaloMETCollection HLT_MET           ,double PtThreshold);
      bool HLTSumEtAbovePtThreshold(const reco::CaloMETCollection HLT_MET           ,double PtThreshold);
      bool HLTJetAbovePtThreshold  (const reco::CaloJetCollection HLT_Jets          ,double PtThreshold);

      bool L1GlobalDecision      (Handle<L1GlobalTriggerReadoutRecord> L1GTRR);
      bool L1GlobalDecision      (bool* TriggerBits);
      bool L1InterestingPath     (int Path);
      bool HLTGlobalDecision     (Handle<TriggerResults> HLTR);
      bool HLTGlobalDecision     (bool* TriggerBits);
      bool HLTInterestingPath    (int Path);
//      bool IsL1ConditionTrue     (int Path,Handle<L1GlobalTriggerReadoutRecord> L1GTRR);
       bool IsL1ConditionTrue(int Path, bool* TriggerBits);


      int ClosestHSCP            (double phi, double eta, double dRMax, const reco::CandidateCollection MC_Cand);
      int ClosestL1Muon          (double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons);
      int ClosestHLTMuon         (double phi, double eta, double dRMax, const reco::RecoChargedCandidateCollection HLT_Muons);

      std::vector<unsigned int> L1OrderByEff();
      std::vector<unsigned int> HLTOrderByEff();

      TFile* Output;

      TH1D*	L1_Mu_Eff_Vs_Thresh_Distrib;
      TH1D*     L1_MET_Eff_Vs_Thresh_Distrib;
      TH1D*     L1_HTT_Eff_Vs_Thresh_Distrib;
      TH1D*     L1_Jet_Eff_Vs_Thresh_Distrib;

      TH1D*     HLT_Mu_Eff_Vs_Thresh_Distrib;
      TH1D*     HLT_MET_Eff_Vs_Thresh_Distrib;
      TH1D*     HLT_SET_Eff_Vs_Thresh_Distrib;
      TH1D*     HLT_Jet_Eff_Vs_Thresh_Distrib;

      TH2D*     L1_MET_Vs_Jet;
  
      TH1D*     L1_Mu_Eff_Vs_MaxBeta_N;
      TH1D*     L1_Mu_Eff_Vs_MaxBeta_Distrib;
      TH1D*     HLT_Mu_Eff_Vs_MaxBeta_N;
      TH1D*     HLT_Mu_Eff_Vs_MaxBeta_Distrib;

      TH1D*     L1_Mu_RecoEff_Vs_Beta_N;
      TH1D*     L1_Mu_RecoEff_Vs_Beta_Distrib;
      TH1D*     HLT_Mu_RecoEff_Vs_Beta_N;
      TH1D*     HLT_Mu_RecoEff_Vs_Beta_Distrib;


      TH1D*     L1_MET_Eff_Vs_MaxBeta_N;
      TH1D*     L1_MET_Eff_Vs_MaxBeta_Distrib;
      TH1D*     HLT_MET_Eff_Vs_MaxBeta_N;
      TH1D*     HLT_MET_Eff_Vs_MaxBeta_Distrib;

      TH1D*     L1_Mu_RecoEff_Vs_eta_N;
      TH1D*     L1_Mu_RecoEff_Vs_eta_Distrib;
      TH1D*     HLT_Mu_RecoEff_Vs_eta_N;
      TH1D*     HLT_Mu_RecoEff_Vs_eta_Distrib;

      TH1D*     HLT_MET_RecoEff_Vs_MaxBeta_N;
      TH1D*     HLT_MET_RecoEff_Vs_MaxBeta_Distrib;
      TH2D*     HLT_MET_RecoEff_Vs_MaxBeta_DPtDPhi;


   std::string   TXT_File_Name;

   std::string*  L1_Names;
   unsigned int* L1_Accepted;
   unsigned int* L1_Rejected;
   unsigned int* L1_Error;
   unsigned int  L1_Global_Accepted;
   unsigned int  L1_Global_Rejected;
   unsigned int  L1_Global_Error;
   unsigned int  L1_NPath;

   std::string*  HLT_Names;
   unsigned int* HLT_WasRun;
   unsigned int* HLT_Accepted;
   unsigned int* HLT_Rejected;
   unsigned int* HLT_Error;
   unsigned int  HLT_Global_Accepted;
   unsigned int  HLT_Global_Rejected;
   unsigned int  HLT_Global_Error;
   unsigned int  HLT_NPath;

   bool		 Init;

   unsigned int  NEvents;
   unsigned int  NEventsBeforeEtaCut;
   unsigned int  NEventsPassL1;

   unsigned int  TableL1_N;
   unsigned int* TableL1_AbsEff;
   unsigned int* TableL1_IncEff;

   unsigned int  TableHLT_N;
   unsigned int* TableHLT_AbsEff;
   unsigned int* TableHLT_IncEff;

   std::vector<unsigned int> TableL1Bis_Sequence;
   unsigned int  TableL1Bis_N;
   unsigned int* TableL1Bis_AbsEff;
   unsigned int* TableL1Bis_IncEff;

   std::vector<unsigned int> TableHLTBis_Sequence;
   unsigned int  TableHLTBis_N;
   unsigned int* TableHLTBis_AbsEff;
   unsigned int* TableHLTBis_IncEff;


   std::vector<bool*>  L1_Trigger_Bits;
   std::vector<bool*> HLT_Trigger_Bits;

   double       DeltaTMax;
   int          recoL1Muon[2];
   double       MinDt[2];
};

HSCP_Trigger::HSCP_Trigger(const edm::ParameterSet& iConfig)

{
      DeltaTMax   = iConfig.getUntrackedParameter<double >("DeltaTMax");

               TXT_File_Name    = iConfig.getUntrackedParameter<std::string>("TextFileName");


   std::string HistoFileName = iConfig.getUntrackedParameter<std::string>("HistoFileName");
   Output = new TFile(HistoFileName.c_str(), "RECREATE");

   L1_Mu_Eff_Vs_Thresh_Distrib     = new TH1D("L1 Muons : Eff Vs Pt Threshold" ,"L1 Muons : Efficiency Vs Pt Threshold" ,200,0,200);
   L1_MET_Eff_Vs_Thresh_Distrib    = new TH1D("L1 MET : Eff Vs Pt Threshold"   ,"L1 MET : Efficiency Vs Pt threshold"   ,250,0,250);
   L1_HTT_Eff_Vs_Thresh_Distrib    = new TH1D("L1 HTT : Eff Vs Pt Threshold"   ,"L1 HTT : Efficiency Vs Pt threshold"   ,500,0,500);
   L1_Jet_Eff_Vs_Thresh_Distrib    = new TH1D("L1 Jet : Eff Vs Pt Threshold"   ,"L1 Jet : Efficiency Vs Pt threshold"   ,250,0,250);

   HLT_Mu_Eff_Vs_Thresh_Distrib    = new TH1D("HLT Muons : Eff Vs Pt Threshold","HLT Muons : Efficiency Vs Pt Threshold",200,0,200);
   HLT_MET_Eff_Vs_Thresh_Distrib   = new TH1D("HLT MET : Eff Vs Pt Threshold"  ,"HLT MET : Efficiency Vs Pt threshold"  ,250,0,250);
   HLT_SET_Eff_Vs_Thresh_Distrib   = new TH1D("HLT SET : Eff Vs Pt Threshold"  ,"HLT SumEt : Efficiency Vs Pt threshold",500,0,500);
   HLT_Jet_Eff_Vs_Thresh_Distrib   = new TH1D("HLT Jet : Eff Vs Pt Threshold"  ,"HLT Jet : Efficiency Vs Pt threshold"  ,250,0,250);

   L1_Mu_Eff_Vs_MaxBeta_N          = new TH1D("L1 Muons : Eff Vs MaxBetaN"     ,"L1 Muons : Efficiency Vs Max HSCP Beta" ,20 ,0,1);
   L1_Mu_Eff_Vs_MaxBeta_Distrib    = new TH1D("L1 Muons : Eff Vs MaxBeta"      ,"L1 Muons : Efficiency Vs Max HSCP Beta" ,20 ,0,1);
   HLT_Mu_Eff_Vs_MaxBeta_N         = new TH1D("HLT Muons : Eff Vs MaxBetaN"    ,"HLT Muons : Efficiency Vs Max HSCP Beta",20 ,0,1);
   HLT_Mu_Eff_Vs_MaxBeta_Distrib   = new TH1D("HLT Muons : Eff Vs MaxBeta"     ,"HLT Muons : Efficiency Vs Max HSCP Beta",20 ,0,1);

   L1_Mu_RecoEff_Vs_Beta_N         = new TH1D("L1 Muons : Reco Eff Vs BetaN"     ,"L1 Muons : Reconstruction Efficiency Vs HSCP Beta" ,20 ,0,1);
   L1_Mu_RecoEff_Vs_Beta_Distrib   = new TH1D("L1 Muons : Reco Eff Vs Beta"      ,"L1 Muons : Reconstruction Efficiency Vs HSCP Beta" ,20 ,0,1);
   HLT_Mu_RecoEff_Vs_Beta_N        = new TH1D("HLT Muons : Reco Eff Vs BetaN"    ,"HLT Muons : Reconstruction Efficiency Vs HSCP Beta",20 ,0,1);
   HLT_Mu_RecoEff_Vs_Beta_Distrib  = new TH1D("HLT Muons : Reco Eff Vs Beta"     ,"HLT Muons : Reconstruction Efficiency Vs HSCP Beta",20 ,0,1);

   L1_MET_Eff_Vs_MaxBeta_N         = new TH1D("L1 MET : Eff Vs MaxBetaN"     ,"L1 MET : Efficiency Vs Max HSCP Beta" ,20 ,0,1);
   L1_MET_Eff_Vs_MaxBeta_Distrib   = new TH1D("L1 MET : Eff Vs MaxBeta"      ,"L1 MET : Efficiency Vs Max HSCP Beta" ,20 ,0,1);
   HLT_MET_Eff_Vs_MaxBeta_N        = new TH1D("HLT MET : Eff Vs MaxBetaN"    ,"HLT MET : Efficiency Vs Max HSCP Beta",20 ,0,1);
   HLT_MET_Eff_Vs_MaxBeta_Distrib  = new TH1D("HLT MET : Eff Vs MaxBeta"     ,"HLT MET : Efficiency Vs Max HSCP Beta",20 ,0,1);

   L1_Mu_RecoEff_Vs_eta_N          = new TH1D("L1 Muons : Reco Eff Vs EtaN"    ,"L1 Muons : Reconstruction Efficiency Vs HSCP Eta" ,11 ,-2.4,2.4);
   L1_Mu_RecoEff_Vs_eta_Distrib    = new TH1D("L1 Muons : Reco Eff Vs Eta"     ,"L1 Muons : Reconstruction Efficiency Vs HSCP Eta" ,11 ,-2.4,2.4);
   HLT_Mu_RecoEff_Vs_eta_N         = new TH1D("HLT Muons : Reco Eff Vs EtaN"   ,"HLT Muons : Reconstruction Efficiency Vs HSCP Eta",11 ,-2.4,2.4);
   HLT_Mu_RecoEff_Vs_eta_Distrib   = new TH1D("HLT Muons : Reco Eff Vs Eta"    ,"HLT Muons : Reconstruction Efficiency Vs HSCP Eta",11 ,-2.4,2.4);

   HLT_MET_RecoEff_Vs_MaxBeta_N         = new TH1D("HLT MET : Reco Eff Vs MaxBetaN"   ,"HLT MET : Reconstruction Efficiency Vs Max HSCP Beta",20 ,0,1);
   HLT_MET_RecoEff_Vs_MaxBeta_Distrib   = new TH1D("HLT MET : Reco Eff Vs MaxBeta"    ,"HLT MET : Reconstruction Efficiency Vs Max HSCP Beta",20 ,0,1);
   HLT_MET_RecoEff_Vs_MaxBeta_DPtDPhi   = new TH2D("HLT MET : Reco Eff Vs MaxBetadPhi","HLT MET : Reconstruction Efficiency Vs Max HSCP Beta",50 ,-50,50,40,-2,2);


   L1_MET_Vs_Jet    		   = new TH2D("L1 HardestJetPt Vs MET"        ,"L1 HardestJetPt Vs MET"                 ,100,0,250,200,0,500);

   L1_Global_Accepted  = 0;
   L1_Global_Rejected  = 0;
   L1_Global_Error     = 0;
   L1_NPath            = 0;
   HLT_Global_Accepted = 0;
   HLT_Global_Rejected = 0;
   HLT_Global_Error    = 0;
   HLT_NPath           = 0;

   Init                = false;

   NEventsBeforeEtaCut = 0;
   NEvents             = 0;
   NEventsPassL1       = 0;

   TableL1Bis_Sequence  = iConfig.getUntrackedParameter<std::vector<unsigned int> >("L1_IncPath_Sequence");
   TableHLTBis_Sequence = iConfig.getUntrackedParameter<std::vector<unsigned int> >("HLT_IncPath_Sequence");

   TableL1_N       = iConfig.getUntrackedParameter<unsigned int>("L1_N_Path_Sequence");
   TableL1_AbsEff  = new unsigned int[TableL1_N+5];
   TableL1_IncEff  = new unsigned int[TableL1_N+5];

   TableL1Bis_N       = TableL1Bis_Sequence.size();
   TableL1Bis_AbsEff  = new unsigned int[TableL1Bis_N+5];
   TableL1Bis_IncEff  = new unsigned int[TableL1Bis_N+5];


   TableHLT_N      = iConfig.getUntrackedParameter<unsigned int>("HLT_N_Path_Sequence");
   TableHLT_AbsEff = new unsigned int[TableHLT_N+5];
   TableHLT_IncEff = new unsigned int[TableHLT_N+5];

   TableHLTBis_N      = TableHLTBis_Sequence.size();
   TableHLTBis_AbsEff = new unsigned int[TableHLTBis_N+5];
   TableHLTBis_IncEff = new unsigned int[TableHLTBis_N+5];


   for(unsigned int i=0;i<TableL1_N+5;i++){
	  TableL1_AbsEff[i]=0;	  TableL1_IncEff[i]=0;     }
   for(unsigned int i=0;i<TableHLT_N+5;i++){
          TableHLT_AbsEff[i]=0;    TableHLT_IncEff[i]=0;   }
   for(unsigned int i=0;i<TableL1Bis_N+5;i++){
          TableL1Bis_AbsEff[i]=0;  TableL1Bis_IncEff[i]=0;     }
   for(unsigned int i=0;i<TableHLTBis_N+5;i++){
          TableHLTBis_AbsEff[i]=0; TableHLTBis_IncEff[i]=0;   }


   L1_Trigger_Bits.clear();
   HLT_Trigger_Bits.clear();
}


HSCP_Trigger::~HSCP_Trigger()
{
   ScaleAndComputeError(L1_Mu_Eff_Vs_Thresh_Distrib ,NEvents);
   ScaleAndComputeError(L1_MET_Eff_Vs_Thresh_Distrib,NEvents);
   ScaleAndComputeError(L1_HTT_Eff_Vs_Thresh_Distrib,NEvents);
   ScaleAndComputeError(L1_Jet_Eff_Vs_Thresh_Distrib,NEvents);

   ScaleAndComputeError(HLT_Mu_Eff_Vs_Thresh_Distrib ,NEvents);
   ScaleAndComputeError(HLT_MET_Eff_Vs_Thresh_Distrib,NEvents);
   ScaleAndComputeError(HLT_SET_Eff_Vs_Thresh_Distrib,NEvents);
   ScaleAndComputeError(HLT_Jet_Eff_Vs_Thresh_Distrib,NEvents);

   DivideAndComputeError(L1_Mu_Eff_Vs_MaxBeta_Distrib,L1_Mu_Eff_Vs_MaxBeta_N);
   delete L1_Mu_Eff_Vs_MaxBeta_N;

   DivideAndComputeError(HLT_Mu_Eff_Vs_MaxBeta_Distrib,HLT_Mu_Eff_Vs_MaxBeta_N);
   delete HLT_Mu_Eff_Vs_MaxBeta_N;

   DivideAndComputeError(L1_MET_Eff_Vs_MaxBeta_Distrib,L1_MET_Eff_Vs_MaxBeta_N);
   delete L1_MET_Eff_Vs_MaxBeta_N;

   DivideAndComputeError(HLT_MET_Eff_Vs_MaxBeta_Distrib,HLT_MET_Eff_Vs_MaxBeta_N);
   delete HLT_MET_Eff_Vs_MaxBeta_N;

   DivideAndComputeError(L1_Mu_RecoEff_Vs_eta_Distrib,L1_Mu_RecoEff_Vs_eta_N);
   delete L1_Mu_RecoEff_Vs_eta_N;

   DivideAndComputeError(HLT_Mu_RecoEff_Vs_eta_Distrib,HLT_Mu_RecoEff_Vs_eta_N);
   delete HLT_Mu_RecoEff_Vs_eta_N;

   DivideAndComputeError(L1_Mu_RecoEff_Vs_Beta_Distrib,L1_Mu_RecoEff_Vs_Beta_N);
   delete L1_Mu_RecoEff_Vs_Beta_N;


   DivideAndComputeError(HLT_Mu_RecoEff_Vs_Beta_Distrib,HLT_Mu_RecoEff_Vs_Beta_N);
   delete HLT_Mu_RecoEff_Vs_Beta_N;


   DivideAndComputeError(HLT_MET_RecoEff_Vs_MaxBeta_Distrib,HLT_MET_RecoEff_Vs_MaxBeta_N);
   delete HLT_MET_RecoEff_Vs_MaxBeta_N;


   Output->Write();
   Output->Close();
}


//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void
HSCP_Trigger::beginJob(const edm::EventSetup&)
{
}

// ------------ method called to for each event  ------------
void
HSCP_Trigger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  Handle<reco::CandidateCollection>  MC_Cand_h ;
  iEvent.getByLabel("genParticleCandidates",MC_Cand_h);
  const reco::CandidateCollection MC_Cand = *MC_Cand_h.product();

  Handle<l1extra::L1JetParticleCollection> L1_CJets_h;
  iEvent.getByLabel("l1extraParticles","Central", L1_CJets_h);
  const l1extra::L1JetParticleCollection L1_CJets = *L1_CJets_h.product();

  Handle<l1extra::L1JetParticleCollection> L1_FJets_h;
  iEvent.getByLabel("l1extraParticles","Forward", L1_FJets_h);
  const l1extra::L1JetParticleCollection L1_FJets = *L1_FJets_h.product();

  Handle<l1extra::L1JetParticleCollection> L1_TJets_h;
  iEvent.getByLabel("l1extraParticles","Tau", L1_TJets_h);
  const l1extra::L1JetParticleCollection L1_TJets = *L1_TJets_h.product();

  l1extra::L1JetParticleCollection L1_Jets;
  for(unsigned int i=0;i<L1_CJets.size();i++){L1_Jets.push_back(L1_CJets[i]);}
  for(unsigned int i=0;i<L1_FJets.size();i++){L1_Jets.push_back(L1_FJets[i]);}
  for(unsigned int i=0;i<L1_TJets.size();i++){L1_Jets.push_back(L1_TJets[i]);}
  std::sort(L1_Jets.begin(), L1_Jets.end(),PtSorter());

  Handle<l1extra::L1EtMissParticle> L1_MET_h;
  iEvent.getByLabel("l1extraParticles", L1_MET_h);
  const l1extra::L1EtMissParticle L1_MET = *L1_MET_h.product();

  Handle<l1extra::L1MuonParticleCollection> L1_Muons_h;
  iEvent.getByLabel("l1extraParticles", L1_Muons_h);
  const l1extra::L1MuonParticleCollection L1_Muons = *L1_Muons_h.product();

  Handle<reco::RecoChargedCandidateCollection>  HLT_Muons_h ;
  try{iEvent.getByLabel("hltL3MuonCandidates",HLT_Muons_h);}catch(...){printf("No hltL3MuonCandidates__HLT\n");}
  reco::RecoChargedCandidateCollection HLT_Muons;
  if(HLT_Muons_h.isValid())HLT_Muons = *HLT_Muons_h.product();

  Handle<reco::CaloJetCollection> HLT_Jets_h;
  InputTag jettag("iterativeCone5CaloJets","","HLT");
  try{iEvent.getByLabel(jettag, HLT_Jets_h);}catch(...){printf("No iterativeCone5CaloJets__HLT\n");}
  reco::CaloJetCollection HLT_Jets;
  if(HLT_Jets_h.isValid())HLT_Jets = *HLT_Jets_h.product();

  Handle<reco::CaloMETCollection> HLT_MET_h;
  InputTag mettag("met","","HLT");
  try{iEvent.getByLabel(mettag, HLT_MET_h);}catch(...){printf("No met__HLT\n");}
  reco::CaloMETCollection HLT_MET;
  if(HLT_MET_h.isValid())HLT_MET = *HLT_MET_h.product();


  bool   CentralEta = false;
  double HSCPbeta   = -1;

  for(unsigned int i=0;i<MC_Cand.size();i++){
	if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 )
        printf("MC  Cand %2i : (%+8.2f,%+8.2f,%+8.2f,%+8.2f)  Pt = %8.2f  PdgId = %+i  Eta = %+6.2f\n",i,MC_Cand[i].energy(), MC_Cand[i].px(),MC_Cand[i].py(),MC_Cand[i].pz(), MC_Cand[i].pt(), MC_Cand[i].pdgId(), MC_Cand[i].eta());
	
	if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 && fabs(MC_Cand[i].eta())<=2.4){
		CentralEta = true;
		if(MC_Cand[i].p()/MC_Cand[i].energy() > HSCPbeta) HSCPbeta = MC_Cand[i].p()/MC_Cand[i].energy();	
	}
  }

  NEventsBeforeEtaCut++;
  if(!CentralEta)return;
  NEvents++;


  
  for(unsigned int i=0;i<L1_Muons.size();i++){
      	printf("L1  Muons %i : (%+8.2f,%+8.2f,%+8.2f,%+8.2f)  Pt = %8.2f  Qual = %i\n",i,L1_Muons[i].energy(), L1_Muons[i].px(),L1_Muons[i].py(),L1_Muons[i].pz(), L1_Muons[i].pt(), L1_Muons[i].gmtMuonCand().quality() );
  }

  for(unsigned int i=0;i<L1_Jets.size();i++){
	printf("L1  Jets %i  : (%+8.2f,%+8.2f,%+8.2f,%+8.2f)  Pt = %8.2f\n",i,L1_Jets[i].energy(), L1_Jets[i].px(),L1_Jets[i].py(),L1_Jets[i].pz(), L1_Jets[i].pt());
  }

  printf("L1  MET     : Pt = %8.2f\n",L1_MET.etMiss());
  printf("L1  HTT     : Pt = %8.2f\n",L1_MET.etHad());


  for(unsigned int i=0;i<HLT_Muons.size();i++){
        printf("HLT Muons %i : (%+8.2f,%+8.2f,%+8.2f,%+8.2f)  Pt = %8.2f\n",i,HLT_Muons[i].energy(), HLT_Muons[i].px(),HLT_Muons[i].py(),HLT_Muons[i].pz(), HLT_Muons[i].pt() );
  }
  for(unsigned int i=0;i<HLT_Jets.size() && i<4;i++){
        printf("HLT Jets %i  : (%+8.2f,%+8.2f,%+8.2f,%+8.2f)  Pt = %8.2f\n",i,HLT_Jets[i].energy(), HLT_Jets[i].px(),HLT_Jets[i].py(),HLT_Jets[i].pz(), HLT_Jets[i].pt());
  }
  for(unsigned int i=0;i<HLT_MET.size();i++){
  	printf("HLT MET     : (%+8.2f,%+8.2f,%+8.2f,%+8.2f)  Pt = %8.2f\n",HLT_MET[i].energy(), HLT_MET[i].px(),HLT_MET[i].py(),HLT_MET[i].pz(), HLT_MET[i].pt());
	printf("HLT SumEt   : %8.2f\n",HLT_MET[i].sumEt());
  }
  


//  int    recoL1Muon[2];
//  double MinDt[2];

  GetTrueL1MuonsAndTime(iEvent, iSetup, recoL1Muon, MinDt);

  for(unsigned int i=0;i<2;i++){
	printf("HSCP %i : ",i);
	if(recoL1Muon[i]>=0){
		printf("Reco L1 Muon = %i with DeltaT = %6.2f ", recoL1Muon[i], MinDt[i]);
		printf("Good Muon = %i",MinDt[i]<DeltaTMax);
	}
        printf("\n");
  }


  //TRIGGER L1 DECISIONS

  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  try {iEvent.getByLabel("l1extraParticleMap",L1GTRR);} catch (...) {;}
//  if (L1GTRR.isValid()) {

	//Initialisation
	if(!Init){
		L1_NPath           = L1GTRR->decisionWord().size();
		L1_Names	   = new std::string [L1_NPath];
		L1_Accepted        = new unsigned int[L1_NPath];
                L1_Rejected 	   = new unsigned int[L1_NPath];
                L1_Error    	   = new unsigned int[L1_NPath];
   		L1_Global_Accepted = 0;
   		L1_Global_Rejected = 0;
  		L1_Global_Error    = 0;

	        for(unsigned int i=0;i<L1_NPath;i++){
        	        l1extra::L1ParticleMap::L1TriggerType type ( static_cast<l1extra::L1ParticleMap::L1TriggerType>(i) );
                	L1_Names[i]    = l1extra::L1ParticleMap::triggerName(type);
                	L1_Accepted[i] = 0; 
	                L1_Rejected[i] = 0;
        	        L1_Error[i]    = 0;
        	} 
	}


        bool* L1_Trigger_Bits_tmp = new bool[L1_NPath];
	for(unsigned int i=0;i<L1_NPath;i++){
		if(i == l1extra::L1ParticleMap::kSingleMu7){
			L1_Trigger_Bits_tmp[i] = L1MuonAbovePtThreshold(L1_Muons,7);			
		}else if(i == l1extra::L1ParticleMap::kDoubleMu3){
                        L1_Trigger_Bits_tmp[i] = L1TwoMuonAbovePtThreshold(L1_Muons,3);	
		}else{
			L1_Trigger_Bits_tmp[i] = L1GTRR->decisionWord()[i];
		}



                if(L1_Trigger_Bits_tmp[i] ){ L1_Accepted[i]++;
                }else{                       L1_Rejected[i]++;}
    	}
        L1_Trigger_Bits.push_back(L1_Trigger_Bits_tmp);

//	if(L1GlobalDecision(L1GTRR)){	L1_Global_Accepted++; 
//	}else{			        L1_Global_Rejected++;}

      if(L1GlobalDecision(L1_Trigger_Bits_tmp)){   L1_Global_Accepted++; 
      }else{                                       L1_Global_Rejected++;}
//  }


  //TRIGGER HLT DECISIONS


   Handle<TriggerResults> HLTR;
   InputTag tag("TriggerResults","","HLT");
   try {iEvent.getByLabel(tag,HLTR);} catch (...) {;}
//   if (HLTR.isValid()) {


        //Initialisation
        if(!Init){
                HLT_NPath          = HLTR->size();
                HLT_Names           = new std::string [HLT_NPath];
                HLT_WasRun          = new unsigned int[HLT_NPath];
                HLT_Accepted        = new unsigned int[HLT_NPath];
                HLT_Rejected        = new unsigned int[HLT_NPath];
                HLT_Error           = new unsigned int[HLT_NPath];
                HLT_Global_Accepted = 0;
                HLT_Global_Rejected = 0;
                HLT_Global_Error    = 0;

		edm::TriggerNames triggerNames;
		triggerNames.init(*HLTR);
                for(unsigned int i=0;i<HLT_NPath;i++){
                        HLT_Names[i]    = (triggerNames.triggerNames() )[i];
	                HLT_WasRun[i]   = 0;
        	        HLT_Accepted[i] = 0;
                	HLT_Rejected[i] = 0;
        	        HLT_Error[i]    = 0;
	        }
		Init = true;
        }


	bool* HLT_Trigger_Bits_tmp = new bool[HLT_NPath];
        for(unsigned int i=0;i<HLT_NPath;i++){
                if(HLTR->wasrun(i) ){ HLT_WasRun  [i]++;}
//                if(HLTR->accept(i)&&IsL1ConditionTrue(i,L1GTRR) ){ HLT_Accepted[i]++;}
                if(HLTR->accept(i)&&IsL1ConditionTrue(i,L1_Trigger_Bits_tmp) ){ HLT_Accepted[i]++;}
                else{                 HLT_Rejected[i]++;}
                if(HLTR->error(i) ) { HLT_Error   [i]++;}

                HLT_Trigger_Bits_tmp[i] = (HLTR->accept(i) && IsL1ConditionTrue(i,L1_Trigger_Bits_tmp));
        }
        HLT_Trigger_Bits.push_back(HLT_Trigger_Bits_tmp);

        if(HLTGlobalDecision(HLTR)){ HLT_Global_Accepted++;
        }else{                       HLT_Global_Rejected++;}
//   }

  bool PassL1 = L1GlobalDecision(L1_Trigger_Bits_tmp);

//  NEvents++;
  if(PassL1) NEventsPassL1++;


  //PLOT DISTRIBUTION L1

  for(int i=0;i<=L1_Mu_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_Mu_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( L1MuonAbovePtThreshold(L1_Muons,BinLowEdge) )
         L1_Mu_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  for(int i=0;i<=L1_MET_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_MET_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( L1METAbovePtThreshold(L1_MET,BinLowEdge))
         L1_MET_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  for(int i=0;i<=L1_HTT_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_HTT_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( L1HTTAbovePtThreshold(L1_MET,BinLowEdge) )
         L1_HTT_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  for(int i=0;i<=L1_Jet_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_Jet_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( L1JetAbovePtThreshold(L1_Jets,BinLowEdge) )
         L1_Jet_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  if(L1_Jets.size()>0)     L1_MET_Vs_Jet->Fill(L1_MET.etMiss(), L1_Jets[0].pt());


  //PLOT DISTRIBUTION HLT

  if(PassL1){
     for(int i=0;i<=HLT_Mu_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_Mu_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HLTMuonAbovePtThreshold(HLT_Muons,BinLowEdge)  && IsL1ConditionTrue(47,L1_Trigger_Bits_tmp) )
            HLT_Mu_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }

     for(int i=0;i<=HLT_MET_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_MET_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HLTMETAbovePtThreshold(HLT_MET,BinLowEdge)  && IsL1ConditionTrue(4,L1_Trigger_Bits_tmp) )
            HLT_MET_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }

     for(int i=0;i<=HLT_SET_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_SET_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HLTSumEtAbovePtThreshold(HLT_MET,BinLowEdge) && IsL1ConditionTrue(12,L1_Trigger_Bits_tmp) )
            HLT_SET_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }

     for(int i=0;i<=HLT_Jet_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_Jet_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HLTJetAbovePtThreshold(HLT_Jets,BinLowEdge) && IsL1ConditionTrue(0,L1_Trigger_Bits_tmp) )
            HLT_Jet_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }
  }


   // MUON BETA DISTRIBUTION

   if(HSCPbeta>=0 && HSCPbeta<1){
       L1_Mu_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
       if(L1MuonAbovePtThreshold(L1_Muons,7) )
          L1_Mu_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);

       if(PassL1){
           HLT_Mu_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
	   if(HLTMuonAbovePtThreshold(HLT_Muons,16) && IsL1ConditionTrue(47,L1_Trigger_Bits_tmp))
              HLT_Mu_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);
       }      
   }

   // MET BETA DISTRIBUTION

   if(HSCPbeta>=0 && HSCPbeta<1){
       L1_MET_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
       if(L1METAbovePtThreshold(L1_MET,30))
          L1_MET_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);

       if(PassL1){
           HLT_MET_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
           if(HLTMETAbovePtThreshold(HLT_MET,65) && IsL1ConditionTrue(4,L1_Trigger_Bits_tmp))
              HLT_MET_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);
       }
   }


   // MUON ETA DISTRIBUTION

  for(unsigned int i=0;i<MC_Cand.size();i++){
        if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 && fabs(MC_Cand[i].eta())<=2.4){
	               L1_Mu_RecoEff_Vs_eta_N->Fill(MC_Cand[i].eta());
	               int I = ClosestL1Muon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,L1_Muons);
                       if( I>=0 && L1_Muons[I].pt()>=7) L1_Mu_RecoEff_Vs_eta_Distrib->Fill(MC_Cand[i].eta());
	   if(PassL1 && IsL1ConditionTrue(47,L1_Trigger_Bits_tmp)){
		       HLT_Mu_RecoEff_Vs_eta_N->Fill(MC_Cand[i].eta());		
                       int I = ClosestHLTMuon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,HLT_Muons);
                       if( I>=0 && HLT_Muons[I].pt()>=16) HLT_Mu_RecoEff_Vs_eta_Distrib->Fill(MC_Cand[i].eta());
           }
        }
  }


   // RECO MUON BETA DISTRIBUTION

  for(unsigned int i=0;i<MC_Cand.size();i++){
        if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 && fabs(MC_Cand[i].eta())<=2.4){
                       L1_Mu_RecoEff_Vs_Beta_N->Fill( MC_Cand[i].p() / MC_Cand[i].energy() );
                       int I = ClosestL1Muon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,L1_Muons);
                       if( I>=0 && L1_Muons[I].pt()>=7) L1_Mu_RecoEff_Vs_Beta_Distrib->Fill(MC_Cand[i].p() / MC_Cand[i].energy());
           if(PassL1 && IsL1ConditionTrue(47,L1_Trigger_Bits_tmp)){
                       HLT_Mu_RecoEff_Vs_Beta_N->Fill(MC_Cand[i].p() / MC_Cand[i].energy());
                       int I = ClosestHLTMuon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,HLT_Muons);
                       if( I>=0 && HLT_Muons[I].pt()>=16) HLT_Mu_RecoEff_Vs_Beta_Distrib->Fill(MC_Cand[i].p() / MC_Cand[i].energy());
           }
        }
  }



   // RECO EFF MET Vs BETA

  if(HSCPbeta>=0 && HSCPbeta<1 && PassL1  && IsL1ConditionTrue(4,L1_Trigger_Bits_tmp) &&  HLT_MET.size()>0){
     reco::Particle::LorentzVector MCMET;
     for(unsigned int i=0;i<MC_Cand.size();i++){
           if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1){
   		MCMET += MC_Cand[i].p4();
           }
     }
     HLT_MET_RecoEff_Vs_MaxBeta_N->Fill(HSCPbeta);
     double dPhi = DeltaPhi(HLT_MET[0].phi(), MCMET.phi() );
     double dE = HLT_MET[0].pt() - MCMET.pt();
     HLT_MET_RecoEff_Vs_MaxBeta_DPtDPhi->Fill(dE,dPhi);
     if(fabs(dPhi)<=0.2 && fabs(dE) <= 30)HLT_MET_RecoEff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);
  }



}


// ------------ method called once each job just after ending the event loop  ------------
void
HSCP_Trigger::endJob() {
  FILE* f = fopen(TXT_File_Name.c_str(),"w");

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@@@@@ ETA CUT @@@@@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"Number of Events\t\t\t\t\t= %i\n",NEventsBeforeEtaCut);
  fprintf(f,"Number of Events with a central HSCP (eta < 2.4)\t= %i\n",NEvents);
  fprintf(f,"Ratio  of Events with a central HSCP (eta < 2.4)\t= %5.2f%%\n",NEvents/(0.01*NEventsBeforeEtaCut));
  fprintf(f,"\n   -->  Trigger Tables are done using only these events\n");



//  FILE* g = fopen("TableName.txt","w");
//  for(unsigned int i=0;i<L1_NPath;i++){
//      fprintf(g,"L1  Path %3i %30s\n",i,L1_Names[i].c_str() );
//  }
//  fprintf(g,"\n\n\n");
//  for(unsigned int i=0;i<HLT_NPath;i++){
//      fprintf(g,"HLT Path %3i %30s\n",i,HLT_Names[i].c_str() );
//  }
//  fclose(g);


  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ L1 TRIGGER SUMMARY @@@@@@@@@@@@@@@@@@@@@@@@\n\n");
  std::vector<unsigned int> L1Ordered = L1OrderByEff();
  for(unsigned int j=0;j<L1Ordered.size();j++){
      unsigned int i = L1Ordered[j];
      fprintf(f,"L1 Path %3i %30s\t: Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",i,L1_Names[i].c_str(),L1_Accepted[i]/(0.01*NEvents),L1_Rejected[i]/(0.01*NEvents),(NEvents-L1_Accepted[i]-L1_Rejected[i])/(0.01*NEvents));
  }

  fprintf(f,"Global Decision\t\t\t\t\t: Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",L1_Global_Accepted/(0.01*NEvents),L1_Global_Rejected/(0.01*NEvents),(NEvents-L1_Global_Accepted-L1_Global_Rejected)/(0.01*NEvents) );



  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ HLT TRIGGER SUMMARY @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  std::vector<unsigned int> HLTOrdered = HLTOrderByEff();
  for(unsigned int j=0;j<HLTOrdered.size();j++){	
	unsigned int i=HLTOrdered[j];	
	if(!HLTInterestingPath(i))continue;
	fprintf(f,"HLT Path %3i %30s\t: Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",i,HLT_Names[i].c_str(),HLT_Accepted[i]/(0.01*NEvents), HLT_Rejected[i]/(0.01*NEvents), HLT_Error[i]/(0.01*NEvents));
  }
  fprintf(f,"HLT Global Decision : NEvent = %6i\t\t  Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",NEvents,HLT_Global_Accepted/(0.01*NEvents), HLT_Global_Rejected/(0.01*NEvents), (NEvents-HLT_Global_Accepted-HLT_Global_Rejected)/(0.01*NEvents) );



  // TABLE COMPUTATION

   for(unsigned int e=0;e<L1_Trigger_Bits.size();e++){

        // TABLE L1

        bool Inc = false;
	for(unsigned int k=0;k<TableL1_N;k++){
           if( (L1_Trigger_Bits[e])[L1Ordered[k]]){                               
                TableL1_AbsEff[k]++;    if(!Inc)TableL1_IncEff[k]++;        Inc = true;
           }
	}

        bool others = false;
        for(unsigned int i=0;i<L1_NPath && others == false;i++){
                if(!L1InterestingPath(i)) continue;		
		bool WrongOthers = false;
		for(unsigned int k=0;k<TableL1_N;k++){WrongOthers = WrongOthers || i==L1Ordered[k];}
		if(!WrongOthers)  others = others || (L1_Trigger_Bits[e])[i];
        }

        if( others ){
                TableL1_AbsEff[TableL1_N]++;    if(!Inc)TableL1_IncEff[TableL1_N]++;        Inc = true;
        }


	bool L1Global = L1GlobalDecision(L1_Trigger_Bits[e]);
        if(  L1Global ){	
                TableL1_AbsEff[TableL1_N+1]++;  if(!Inc)TableL1_IncEff[TableL1_N+1]++;      Inc = true;
        }



        // TABLE L1 Bis

        Inc = false;
        for(unsigned int k=0;k<TableL1Bis_N;k++){
           if( (L1_Trigger_Bits[e])[TableL1Bis_Sequence[k]]){
                TableL1Bis_AbsEff[k]++;    if(!Inc)TableL1Bis_IncEff[k]++;        Inc = true;
           }
        }

        others = false;
        for(unsigned int i=0;i<L1_NPath && others == false;i++){
                if(!L1InterestingPath(i)) continue;
                bool WrongOthers = false;
                for(unsigned int k=0;k<TableL1Bis_N;k++){WrongOthers = WrongOthers || i==TableL1Bis_Sequence[k];}
                if(!WrongOthers)  others = others || (L1_Trigger_Bits[e])[i];
        }

        if( others ){
                TableL1Bis_AbsEff[TableL1Bis_N]++; if(!Inc)TableL1Bis_IncEff[TableL1Bis_N]++;  Inc = true;
        }


        if(  L1Global ){
                TableL1Bis_AbsEff[TableL1Bis_N+1]++;if(!Inc)TableL1Bis_IncEff[TableL1Bis_N+1]++;Inc = true;
        }


        // TABLE HLT

	bool tot = false;
        Inc  = false;
        for(unsigned int k=0;k<TableHLT_N;k++){
           if( (HLT_Trigger_Bits[e])[HLTOrdered[k]]){
                TableHLT_AbsEff[k]++;    if(!Inc)TableHLT_IncEff[k]++;        Inc = true;
           }
 	   tot = tot || (HLT_Trigger_Bits[e])[HLTOrdered[k]];
        }

        if(tot){
                TableHLT_AbsEff[TableHLT_N]++;
        }

        others = false;
        for(unsigned int i=0;i<HLT_NPath && others == false;i++){
                if(!HLTInterestingPath(i)) continue;
                bool WrongOthers = false;
                for(unsigned int k=0;k<TableHLT_N;k++){WrongOthers = WrongOthers || i==HLTOrdered[k];}
                if(!WrongOthers)  others = others || (HLT_Trigger_Bits[e])[i];
        }

        if( others ){
                TableHLT_AbsEff[TableHLT_N+1]++;if(!Inc)TableHLT_IncEff[TableHLT_N+1]++;  Inc = true;
        }

        // TABLE HLT Bis

        tot = false;
        Inc = false;
        for(unsigned int k=0;k<TableHLTBis_N;k++){
           if( (HLT_Trigger_Bits[e])[TableHLTBis_Sequence[k]]){
                TableHLTBis_AbsEff[k]++;  if(!Inc)TableHLTBis_IncEff[k]++;      Inc = true;
           }
           tot = tot || (HLT_Trigger_Bits[e])[TableHLTBis_Sequence[k]];
        }

        if(tot){
                TableHLTBis_AbsEff[TableHLTBis_N]++;
        }

        others = false;
        for(unsigned int i=0;i<HLT_NPath && others == false;i++){
                if(!HLTInterestingPath(i)) continue;
                bool WrongOthers = false;
                for(unsigned int k=0;k<TableHLTBis_N;k++){WrongOthers = WrongOthers || i==TableHLTBis_Sequence[k];}
                if(!WrongOthers)  others = others || (HLT_Trigger_Bits[e])[i];
        }

        if( others ){
                TableHLTBis_AbsEff[TableHLTBis_N+1]++;if(!Inc)TableHLTBis_IncEff[TableHLTBis_N+1]++;  Inc = true;
        }
  }



  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ L1  TRIGGER TABLE  @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%20s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableL1_N;k++){
     fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n",L1_Names[L1Ordered[k]].c_str(),TableL1_AbsEff[k]/(0.01*NEvents), TableL1_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","L1_Others"   ,TableL1_AbsEff[TableL1_N]/(0.01*NEvents), TableL1_IncEff[TableL1_N]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","L1_Total"    ,TableL1_AbsEff[TableL1_N+1]/(0.01*NEvents), TableL1_IncEff[TableL1_N+1]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ HLT TRIGGER TABLE @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%20s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableHLT_N;k++){
      fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n",HLT_Names[HLTOrdered[k]].c_str(),TableHLT_AbsEff[k]/(0.01*NEvents), TableHLT_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","HLT_Previous_Paths",TableHLT_AbsEff[TableHLT_N]/(0.01*NEvents), TableHLT_IncEff[TableHLT_N]/(0.01*NEvents));
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","HLT_Others"        ,TableHLT_AbsEff[TableHLT_N+1]/(0.01*NEvents), TableHLT_IncEff[TableHLT_N+1]/(0.01*NEvents)
);
  fprintf(f,"----------------------------------------------------------\n");


  if(TableL1Bis_N>0){
  
  fprintf(f,"\n\n\n<><><> Personal Trigger Path Sequence <><><>\n\n\n");

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ L1  PERSONAL TRIGGER TABLE  @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%20s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableL1Bis_N;k++){
     fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n",L1_Names[TableL1Bis_Sequence[k]].c_str(),TableL1Bis_AbsEff[k]/(0.01*NEvents), TableL1Bis_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","L1_Others"   ,TableL1Bis_AbsEff[TableL1Bis_N]/(0.01*NEvents), TableL1Bis_IncEff[TableL1Bis_N]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","L1_Total"    ,TableL1Bis_AbsEff[TableL1Bis_N+1]/(0.01*NEvents), TableL1Bis_IncEff[TableL1Bis_N+1]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");

  }


  if( TableHLTBis_N >0){

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ HLT PERSONAL TRIGGER TABLE @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%20s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableHLTBis_N;k++){
      fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n",HLT_Names[TableHLTBis_Sequence[k]].c_str(),TableHLTBis_AbsEff[k]/(0.01*NEvents), TableHLTBis_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","HLT_Previous_Paths",TableHLTBis_AbsEff[TableHLTBis_N]/(0.01*NEvents), TableHLTBis_IncEff[TableHLTBis_N]/(0.01*NEvents));
  fprintf(f,"%20s |     %6.2f%%     |     %6.2f%%\n","HLT_Others"        ,TableHLTBis_AbsEff[TableHLTBis_N+1]/(0.01*NEvents), TableHLTBis_IncEff[TableHLTBis_N+1]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");



  }

  fclose(f);
}


bool
HSCP_Trigger::L1MuonAbovePtThreshold(const l1extra::L1MuonParticleCollection L1_Muons, double PtThreshold)
{
  for(unsigned int i=0;i<L1_Muons.size();i++){
        if(recoL1Muon[0]==(int)i && MinDt[0] >= DeltaTMax) continue;
        if(recoL1Muon[1]==(int)i && MinDt[1] >= DeltaTMax) continue;
        if(L1_Muons[i].gmtMuonCand().quality()>=4 && L1_Muons[i].et() >= PtThreshold)return true;}
  return false;
}


bool
HSCP_Trigger::L1TwoMuonAbovePtThreshold(const l1extra::L1MuonParticleCollection L1_Muons, double PtThreshold)
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
HSCP_Trigger::L1METAbovePtThreshold(const l1extra::L1EtMissParticle L1_MET, double PtThreshold)
{
  if(L1_MET.etMiss() >= PtThreshold)return true;
  return false;
}

bool
HSCP_Trigger::L1HTTAbovePtThreshold(const l1extra::L1EtMissParticle L1_MET, double PtThreshold)
{
  if(L1_MET.etHad() >= PtThreshold)return true;
  return false;
}


bool
HSCP_Trigger::L1JetAbovePtThreshold(const l1extra::L1JetParticleCollection L1_Jets, double PtThreshold)
{
  for(unsigned int i=0;i<L1_Jets.size();i++){
        if(L1_Jets[i].et() >= PtThreshold)return true; }  
  return false;
}

bool
HSCP_Trigger::HLTMuonAbovePtThreshold(const  reco::RecoChargedCandidateCollection HLT_Muons, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_Muons.size();i++){
        if(HLT_Muons[i].et() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger::HLTMETAbovePtThreshold(const reco::CaloMETCollection HLT_MET, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_MET.size();i++){
	  if(HLT_MET[i].et() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger::HLTSumEtAbovePtThreshold(const reco::CaloMETCollection HLT_MET, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_MET.size();i++){
          if(HLT_MET[i].sumEt() >= PtThreshold)return true;}
  return false;
}

bool
HSCP_Trigger::HLTJetAbovePtThreshold(const reco::CaloJetCollection HLT_Jets, double PtThreshold)
{
  for(unsigned int i=0;i<HLT_Jets.size();i++){
        if(HLT_Jets[i].et() >= PtThreshold)return true; }
  return false;
}



bool
HSCP_Trigger::L1GlobalDecision(bool* TriggerBits)
{
   for(unsigned int i=0;i<L1_NPath;i++){
        if(L1InterestingPath(i) && TriggerBits[i]) return true;
   }
   return false;
}


/*
bool
HSCP_Trigger::L1GlobalDecision(Handle<L1GlobalTriggerReadoutRecord> L1GTRR)
{
  if (L1GTRR.isValid()) {
	for(unsigned int i=0;i<L1_NPath;i++)
	{
		if(L1InterestingPath(i) && L1GTRR->decisionWord()[i]) return true;
	}
  }
  return false;
}
*/

bool 
HSCP_Trigger::L1InterestingPath(int Path)
{
    if(Path == l1extra::L1ParticleMap::kSingleMu7)                return true;
    if(Path == l1extra::L1ParticleMap::kSingleIsoEG12)            return true;
    if(Path == l1extra::L1ParticleMap::kSingleEG15)               return true;
    if(Path == l1extra::L1ParticleMap::kSingleJet100)             return true;
    if(Path == l1extra::L1ParticleMap::kSingleTauJet80)           return true;
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
    if(Path == l1extra::L1ParticleMap::kExclusiveDoubleJet60)     return true;
    if(Path == l1extra::L1ParticleMap::kExclusiveJet25_Gap_Jet25) return true;
    if(Path == l1extra::L1ParticleMap::kIsoEG10_Jet20_ForJet10)   return true;

    return false;
}





bool
HSCP_Trigger::HLTGlobalDecision(bool* TriggerBits)
{
   for(unsigned int i=0;i<HLT_NPath;i++){
        if(HLTInterestingPath(i) && TriggerBits[i]) return true;
   }
   return false;
}



bool
HSCP_Trigger::HLTGlobalDecision(Handle<TriggerResults> HLTR)
{
   if (HLTR.isValid()) {
        for(unsigned int i=0;i<HLT_NPath;i++){                
                if(HLTInterestingPath(i) && HLTR->accept(i)) return true;
	}
   }
   return false;
}


bool
//HSCP_Trigger::IsL1ConditionTrue(int Path,Handle<L1GlobalTriggerReadoutRecord> L1GTRR)
HSCP_Trigger::IsL1ConditionTrue(int Path, bool* TriggerBits)
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
HSCP_Trigger::HLTInterestingPath(int Path)
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


std::vector<unsigned int>
HSCP_Trigger::L1OrderByEff(){
  std::vector<unsigned int> To_return;
  unsigned int max = 0;  int I=-1;
  do{      
      max = 0;  I=-1;
      for(unsigned int i=0;i<L1_NPath;i++){
          if(!L1InterestingPath(i)) continue;
          bool AlreadyIn = false;
          for(unsigned int j=0;j<To_return.size();j++){AlreadyIn = AlreadyIn || (i==To_return[j]);}
          if(!AlreadyIn && L1_Accepted[i]>=max){ max = L1_Accepted[i];	I =i; }
      }
      if(I!=-1)To_return.push_back(I);
  }while(I!=-1);
  return To_return;
}

std::vector<unsigned int>
HSCP_Trigger::HLTOrderByEff(){
  std::vector<unsigned int> To_return;
  unsigned int max = 0;  int I=-1;
  do{
      max = 0;  I=-1;
      for(unsigned int i=0;i<HLT_NPath;i++){
          if(!HLTInterestingPath(i)) continue;
          bool AlreadyIn = false;
          for(unsigned int j=0;j<To_return.size();j++){AlreadyIn = AlreadyIn || (i==To_return[j]);}
          if(!AlreadyIn && HLT_Accepted[i]>=max){ max = HLT_Accepted[i];  I =i; }
      }
      if(I!=-1)To_return.push_back(I);
  }while(I!=-1);
  return To_return;
}



int
HSCP_Trigger::ClosestHSCP(double phi, double eta, double dRMax, const reco::CandidateCollection MC_Cand)
{
       double dR = 99999; int J=-1;
       for(unsigned int j=0;j<MC_Cand.size();j++){
          if(dR > DeltaR(phi,eta, MC_Cand[j].phi(), MC_Cand[j].eta()) ){
             dR = DeltaR(phi,eta, MC_Cand[j].phi(), MC_Cand[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;	
}

int
HSCP_Trigger::ClosestL1Muon(double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons)
{
       double dR = 99999; int J=-1;
       for(unsigned int j=0;j<L1_Muons.size();j++){
          if(dR > DeltaR(phi,eta, L1_Muons[j].phi(), L1_Muons[j].eta()) ){
             dR = DeltaR(phi,eta, L1_Muons[j].phi(), L1_Muons[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;
}

int
HSCP_Trigger::ClosestHLTMuon(double phi, double eta, double dRMax, const  reco::RecoChargedCandidateCollection HLT_Muons)
{
       double dR = 99999; int J=-1;
       for(unsigned int j=0;j<HLT_Muons.size();j++){
          if(dR > DeltaR(phi,eta, HLT_Muons[j].phi(), HLT_Muons[j].eta()) ){
             dR = DeltaR(phi,eta, HLT_Muons[j].phi(), HLT_Muons[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;
}



double
DeltaR(double phi1, double eta1, double phi2, double eta2)
{
        double deltaphi=phi1-phi2;

        if(fabs(deltaphi)>3.14)deltaphi=2*3.14-fabs(deltaphi);
        else if(fabs(deltaphi)<3.14)deltaphi=fabs(deltaphi);
        return (sqrt(pow(deltaphi,2)+pow(eta1 - eta2,2)));
}

double
DeltaPhi(double phi1,  double phi2)
{
        double deltaphi=phi1-phi2;

        if(fabs(deltaphi)>3.14)deltaphi=2*3.14-fabs(deltaphi);
        else if(fabs(deltaphi)<3.14)deltaphi=fabs(deltaphi);
        return deltaphi;
}

void
DivideAndComputeError(TH1D* Distrib, TH1D* N)
{
  Distrib->Divide(N);
  for(int i=0;i<=Distrib->GetNbinsX();i++){
     if(N->GetBinContent(i) > 1)
     {
        double eff = Distrib->GetBinContent(i);
        double err = sqrt( eff*(1-eff) / N->GetBinContent(i));
//        if(eff==0 || eff==1) err = 1/sqrt( N->GetBinContent(i) );
        Distrib->SetBinError  (i,err);
     }else{
        Distrib->SetBinContent(i,0);
        Distrib->SetBinError  (i,0);
     }
  }
}

void
ScaleAndComputeError(TH1D* Distrib, unsigned int N)
{
  Distrib->Scale(1.0/N);
  for(int i=0;i<=Distrib->GetNbinsX();i++){
    double eff = Distrib->GetBinContent(i);
    double err = sqrt( eff*(1-eff) / N);
//      if(eff==0 || eff==1) err = 1/sqrt( N->GetBinContent(i) );
    Distrib->SetBinError  (i,err);
  }
}


char* LatexFormat(const char* txt)
{
  char* output = new char[255];
  unsigned int j =0;
  for(unsigned int i=0; j<250 && txt[i]!='\0' ; i++){
     if(txt[i] == '_'){ output[j]='\\';j++;}   
     output[j] = txt[i];
     j++; 
  }
  output[j] = '\0';
  return output;
}




//define this as a plug-in
DEFINE_FWK_MODULE(HSCP_Trigger);




