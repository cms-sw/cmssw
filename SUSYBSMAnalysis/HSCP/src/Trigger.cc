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
// $Id: HSCP_Trigger.cc,v 1.5 2008/01/21 10:39:10 querten Exp $
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

#include "SUSYBSMAnalysis/HSCP/interface/Trigger_MainFunctions.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"


#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

using namespace edm;

void   DivideAndComputeError(TH1D* Distrib, TH1D* N);
void   ScaleAndComputeError(TH1D* Distrib, unsigned int N);

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

      std::vector<unsigned int> L1OrderByEff();
      std::vector<unsigned int> L1OrderByIncEff();
      std::vector<unsigned int> HLTOrderByEff();
      std::vector<unsigned int> HLTOrderByIncEff();

      int GetL1Path(const char* name);
      int GetHLTPath(const char* name);


      TFile* Output;

      TH1D*	L1_1Mu_Eff_Vs_Thresh_Distrib;
      TH1D*     L1_2Mu_Eff_Vs_Thresh_Distrib;
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

      TH2D*     HLT_Muon_Eta_Vs_Beta;
      TH2D*     HLT_Muon_Eta_Vs_BetaBefCut;
      TH2D*     L1_Muon_Eta_Vs_Beta;
      TH2D*     L1_Muon_Eta_Vs_BetaBefCut;
      TH2D*     MC_Muon_Eta_Vs_Beta;


      double EtaOfHSCP;
      bool   UseOnlyL1MuonInBarel;

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

   std::vector<std::string> TableL1Bis_Sequence;
   unsigned int  TableL1Bis_N;
   unsigned int* TableL1Bis_AbsEff;
   unsigned int* TableL1Bis_IncEff;

   std::vector<std::string> TableHLTBis_Sequence;
   unsigned int  TableHLTBis_N;
   unsigned int* TableHLTBis_AbsEff;
   unsigned int* TableHLTBis_IncEff;


   std::vector<bool*>  L1_Trigger_Bits;
   std::vector<bool*> HLT_Trigger_Bits;
};

HSCP_Trigger::HSCP_Trigger(const edm::ParameterSet& iConfig)

{
   TH1::AddDirectory(kTRUE);

   TXT_File_Name    = iConfig.getUntrackedParameter<std::string>("TextFileName");

   std::string HistoFileName = iConfig.getUntrackedParameter<std::string>("HistoFileName");
   Output = new TFile(HistoFileName.c_str(), "RECREATE");
   Output->cd();

   L1_1Mu_Eff_Vs_Thresh_Distrib    = new TH1D("L1 1Muons : Eff Vs Pt Threshold" ,"L1 1Muons : Efficiency Vs Pt Threshold" ,200,0,200);
   L1_2Mu_Eff_Vs_Thresh_Distrib    = new TH1D("L1 2Muons : Eff Vs Pt Threshold" ,"L1 2Muons : Efficiency Vs Pt Threshold" ,200,0,200);
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

   L1_Mu_RecoEff_Vs_eta_N          = new TH1D("L1 Muons : Reco Eff Vs EtaN"    ,"L1 Muons : Reconstruction Efficiency Vs HSCP Eta" ,11 ,-2.1,2.1);
   L1_Mu_RecoEff_Vs_eta_Distrib    = new TH1D("L1 Muons : Reco Eff Vs Eta"     ,"L1 Muons : Reconstruction Efficiency Vs HSCP Eta" ,11 ,-2.1,2.1);
   HLT_Mu_RecoEff_Vs_eta_N         = new TH1D("HLT Muons : Reco Eff Vs EtaN"   ,"HLT Muons : Reconstruction Efficiency Vs HSCP Eta",11 ,-2.1,2.1);
   HLT_Mu_RecoEff_Vs_eta_Distrib   = new TH1D("HLT Muons : Reco Eff Vs Eta"    ,"HLT Muons : Reconstruction Efficiency Vs HSCP Eta",11 ,-2.1,2.1);

   HLT_MET_RecoEff_Vs_MaxBeta_N         = new TH1D("HLT MET : Reco Eff Vs MaxBetaN"   ,"HLT MET : Reconstruction Efficiency Vs Max HSCP Beta",20 ,0,1);
   HLT_MET_RecoEff_Vs_MaxBeta_Distrib   = new TH1D("HLT MET : Reco Eff Vs MaxBeta"    ,"HLT MET : Reconstruction Efficiency Vs Max HSCP Beta",20 ,0,1);
   HLT_MET_RecoEff_Vs_MaxBeta_DPtDPhi   = new TH2D("HLT MET : Reco Eff Vs MaxBetadPhi","HLT MET : Reconstruction Efficiency Vs Max HSCP Beta",50 ,-50,50,40,-2,2);


   L1_MET_Vs_Jet    		   = new TH2D("L1 HardestJetPt Vs MET"        ,"L1 HardestJetPt Vs MET"                 ,100,0,250,200,0,500);

   HLT_Muon_Eta_Vs_Beta 	= new TH2D("HLT Muon Eta Vs Beta"       ,"HLT Muon Eta Vs Beta"       ,21,-2.1,2.1,50,0,1);
   HLT_Muon_Eta_Vs_BetaBefCut 	= new TH2D("HLT Muon Eta Vs BetaBefCut" ,"HLT Muon Eta Vs BetaBefCut" ,21,-2.1,2.1,50,0,1);
   L1_Muon_Eta_Vs_Beta  	= new TH2D("L1 Muon Eta Vs Beta"        ,"L1 Muon Eta Vs Beta"        ,21,-2.1,2.1,50,0,1); 
   L1_Muon_Eta_Vs_BetaBefCut  	= new TH2D("L1 Muon Eta Vs BetaBefCut"  ,"L1 Muon Eta Vs BetaBefCut"  ,21,-2.1,2.1,50,0,1);
   MC_Muon_Eta_Vs_Beta  	= new TH2D("MC HSCP Eta Vs Beta"        ,"MC HSCP Eta Vs Beta"        ,21,-2.1,2.1,50,0,1);

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

   EtaOfHSCP = iConfig.getUntrackedParameter<double>("AtLeastOneHSCPInEta");

   TableL1Bis_Sequence  = iConfig.getUntrackedParameter<std::vector<std::string> >("L1_IncPath_Sequence");
   TableHLTBis_Sequence = iConfig.getUntrackedParameter<std::vector<std::string> >("HLT_IncPath_Sequence");

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
   ScaleAndComputeError(L1_1Mu_Eff_Vs_Thresh_Distrib ,NEvents);
   ScaleAndComputeError(L1_2Mu_Eff_Vs_Thresh_Distrib ,NEvents);
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

  Handle<l1extra::L1EtMissParticleCollection> L1_MET_h;
  iEvent.getByLabel("l1extraParticles", L1_MET_h);
  const l1extra::L1EtMissParticleCollection L1_MET = *L1_MET_h.product();

  Handle<l1extra::L1MuonParticleCollection> L1_Muons_h;
  iEvent.getByLabel("l1extraParticles", L1_Muons_h);
  const l1extra::L1MuonParticleCollection L1_MuonsT = *L1_Muons_h.product();

  l1extra::L1MuonParticleCollection L1_Muons;
  for(unsigned int i=0;i<L1_MuonsT.size();i++){
        L1_Muons.push_back(L1_MuonsT[i]);
  }

  Handle<reco::RecoChargedCandidateCollection>  HLT_Muons_h ;
  InputTag muontag("hltL3MuonCandidates","","HLT");
  try{iEvent.getByLabel(muontag,HLT_Muons_h);}catch(...){printf("No hltL3MuonCandidates__HLT\n");}
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
	
	if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 && fabs(MC_Cand[i].eta())<=EtaOfHSCP){
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

  printf("L1  MET     : Pt = %8.2f\n",L1_MET[0].etMiss());
  printf("L1  HTT     : Pt = %8.2f\n",L1_MET[0].etHad());


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
  

  //TRIGGER L1 DECISIONS  //CAN BE IMPROVE USING :  CMSSW/L1Trigger/GlobalTriggerAnalyzer/src/L1GtTrigReport.cc

  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  try {iEvent.getByLabel("l1GtEmulDigis",L1GTRR);} catch (...) {;}
  if(!L1GTRR.isValid()){cout<<"L1GlobalTriggerReadoutRecord with Label l1GtEmulDigis is not Valid" << endl;exit(0);}

//  Handle<edm::Handle<L1GlobalTriggerObjectMapRecord> L1GTOMR;
//  try {iEvent.getByLabel("l1GtEmulDigis",L1GTOMR);} catch (...) {;}
//  if(!L1GTOMR.isValid()){cout<<"L1GlobalTriggerObjectMapRecord with Label l1GtEmulDigis is not Valid" << endl;exit(0);}

    edm::ESHandle< L1GtTriggerMenu> l1GtMenu;
    iSetup.get< L1GtTriggerMenuRcd>().get(l1GtMenu) ;

    AlgorithmMap algorithmMap = l1GtMenu->gtAlgorithmMap();
    std::string menuName = l1GtMenu->gtTriggerMenuName();


  //Initialisation
  if(!Init){
//	L1_NPath           = L1GTRR->decisionWord().size() -5; //BUG IF REMOVE -5 --> WHY ????
        L1_NPath           = algorithmMap.size();
	L1_Names	   = new std::string [L1_NPath];
	L1_Accepted        = new unsigned int[L1_NPath];
        L1_Rejected 	   = new unsigned int[L1_NPath];
        L1_Error    	   = new unsigned int[L1_NPath];
	L1_Global_Accepted = 0;
	L1_Global_Rejected = 0;
	L1_Global_Error    = 0;

        for(unsigned int i=0;i<L1_NPath;i++){
                L1_Names[i]    = "unknown";
                L1_Accepted[i] = 0;
                L1_Rejected[i] = 0;
                L1_Error[i]    = 0;
        }


        for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
                unsigned int i = (itAlgo->second)->algoBitNumber();
                L1_Names[i]    = itAlgo->first;
                printf("%2i --> %s\n",i,itAlgo->first.c_str());
        }
   }

   bool PassL1 = false;
   bool* L1_Trigger_Bits_tmp = new bool[L1_NPath];
   for(unsigned int i=0;i<L1_NPath;i++){
	L1_Trigger_Bits_tmp[i] = L1GTRR->decisionWord()[i];
        if(L1_Trigger_Bits_tmp[i] ){ L1_Accepted[i]++; PassL1=true;
        }else{                       L1_Rejected[i]++;}
   }

   L1_Trigger_Bits.push_back(L1_Trigger_Bits_tmp);

   if(PassL1){
	L1_Global_Accepted++;  NEventsPassL1++;
   }else{
 	L1_Global_Rejected++;
   }

  //TRIGGER HLT DECISIONS

   Handle<TriggerResults> HLTR;
   InputTag tag("TriggerResults","","HLT");
   try {iEvent.getByLabel(tag,HLTR);} catch (...) {;}

   //Initialisation
   if(!Init){
        HLT_NPath           = HLTR->size();
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
       HLT_Trigger_Bits_tmp[i] = HLTR->accept(i);       

       if(HLTR->wasrun(i)        ){ HLT_WasRun  [i]++;}
       if(HLT_Trigger_Bits_tmp[i]){ HLT_Accepted[i]++;}
       else{                        HLT_Rejected[i]++;}
       if(HLTR->error(i)         ){ HLT_Error   [i]++;}
   }
   HLT_Trigger_Bits.push_back(HLT_Trigger_Bits_tmp);

  if( HLTR->accept()){ HLT_Global_Accepted++;
  }else{               HLT_Global_Rejected++;}


  for(int i=0;i<=L1_1Mu_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_1Mu_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( HSCP_Trigger_L1MuonAbovePtThreshold(L1_Muons,BinLowEdge) )
         L1_1Mu_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  for(int i=0;i<=L1_2Mu_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_2Mu_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( HSCP_Trigger_L1TwoMuonAbovePtThreshold(L1_Muons,BinLowEdge) )
         L1_2Mu_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }


  for(int i=0;i<=L1_MET_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_MET_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( HSCP_Trigger_L1METAbovePtThreshold(L1_MET[0],BinLowEdge))
         L1_MET_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  for(int i=0;i<=L1_HTT_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_HTT_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( HSCP_Trigger_L1HTTAbovePtThreshold(L1_MET[0],BinLowEdge) )
         L1_HTT_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }

  for(int i=0;i<=L1_Jet_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
     double BinLowEdge = L1_Jet_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
     if( HSCP_Trigger_L1JetAbovePtThreshold(L1_Jets,BinLowEdge) )
         L1_Jet_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
  }
  if(L1_Jets.size()>0)     L1_MET_Vs_Jet->Fill(L1_MET[0].etMiss(), L1_Jets[0].pt());


  //PLOT DISTRIBUTION HLT

  if(PassL1){
     for(int i=0;i<=HLT_Mu_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_Mu_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HSCP_Trigger_HLTMuonAbovePtThreshold(HLT_Muons,BinLowEdge) && L1_Trigger_Bits_tmp[GetL1Path("L1_SingleMu7")] )
            HLT_Mu_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }

     for(int i=0;i<=HLT_MET_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_MET_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HSCP_Trigger_HLTMETAbovePtThreshold(HLT_MET,BinLowEdge) && L1_Trigger_Bits_tmp[GetL1Path("L1_ETM40")] )
            HLT_MET_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }

     for(int i=0;i<=HLT_SET_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_SET_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HSCP_Trigger_HLTSumEtAbovePtThreshold(HLT_MET,BinLowEdge) && L1_Trigger_Bits_tmp[GetL1Path("L1_HTT300")] )  //HTT200 in 167
            HLT_SET_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }

     for(int i=0;i<=HLT_Jet_Eff_Vs_Thresh_Distrib->GetNbinsX();i++){
        double BinLowEdge = HLT_Jet_Eff_Vs_Thresh_Distrib->GetBinLowEdge(i);
        if( HSCP_Trigger_HLTJetAbovePtThreshold(HLT_Jets,BinLowEdge) && L1_Trigger_Bits_tmp[GetL1Path("L1_SingleJet150")] )
            HLT_Jet_Eff_Vs_Thresh_Distrib->Fill(BinLowEdge);
     }
  }


   // MUON BETA DISTRIBUTION

   if(HSCPbeta>=0 && HSCPbeta<1){
       L1_Mu_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
       if(HSCP_Trigger_L1MuonAbovePtThreshold(L1_Muons,7) )
          L1_Mu_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);

       if(PassL1){
           HLT_Mu_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
	   if(HSCP_Trigger_HLTMuonAbovePtThreshold(HLT_Muons,16) && L1_Trigger_Bits_tmp[GetL1Path("L1_SingleMu7")] )
              HLT_Mu_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);
       }      
   }

   // MET BETA DISTRIBUTION

   if(HSCPbeta>=0 && HSCPbeta<1){
       L1_MET_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
       if(HSCP_Trigger_L1METAbovePtThreshold(L1_MET[0],30))
          L1_MET_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);

       if(PassL1){
           HLT_MET_Eff_Vs_MaxBeta_N->Fill(HSCPbeta);
           if(HSCP_Trigger_HLTMETAbovePtThreshold(HLT_MET,65) && L1_Trigger_Bits_tmp[GetL1Path("L1_ETM40")] )
              HLT_MET_Eff_Vs_MaxBeta_Distrib->Fill(HSCPbeta);
       }
   }


   // MUON ETA DISTRIBUTION

  for(unsigned int i=0;i<MC_Cand.size();i++){
        if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 && fabs(MC_Cand[i].eta())<=2.4){
	               L1_Mu_RecoEff_Vs_eta_N->Fill(MC_Cand[i].eta());

		       MC_Muon_Eta_Vs_Beta->Fill(MC_Cand[i].eta(), MC_Cand[i].p()/MC_Cand[i].energy());


	               int I = HSCP_Trigger_ClosestL1Muon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,L1_Muons);
		       if(I<0)continue;
                        L1_Muon_Eta_Vs_BetaBefCut->Fill(MC_Cand[i].eta(), MC_Cand[i].p()/MC_Cand[i].energy());
                       if(L1_Muons[I].pt()>=7) L1_Muon_Eta_Vs_Beta->Fill(MC_Cand[i].eta(), MC_Cand[i].p()/MC_Cand[i].energy());
                       if(L1_Muons[I].pt()>=7) L1_Mu_RecoEff_Vs_eta_Distrib->Fill(MC_Cand[i].eta());


	   if(PassL1 && L1_Trigger_Bits_tmp[GetL1Path("L1_SingleMu7")]){
		       HLT_Mu_RecoEff_Vs_eta_N->Fill(MC_Cand[i].eta());		
                       int I = HSCP_Trigger_ClosestHLTMuon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,HLT_Muons);
                       if( I>=0 ) HLT_Muon_Eta_Vs_BetaBefCut->Fill(MC_Cand[i].eta(), MC_Cand[i].p()/MC_Cand[i].energy());
		       if( I>=0 && HLT_Muons[I].pt()>=16) HLT_Muon_Eta_Vs_Beta->Fill(MC_Cand[i].eta(), MC_Cand[i].p()/MC_Cand[i].energy());
                       if( I>=0 && HLT_Muons[I].pt()>=16) HLT_Mu_RecoEff_Vs_eta_Distrib->Fill(MC_Cand[i].eta());

           }
        }
  }


   // RECO MUON BETA DISTRIBUTION

  for(unsigned int i=0;i<MC_Cand.size();i++){
        if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 && fabs(MC_Cand[i].eta())<=2.4){
                       L1_Mu_RecoEff_Vs_Beta_N->Fill( MC_Cand[i].p() / MC_Cand[i].energy() );
                       int I = HSCP_Trigger_ClosestL1Muon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,L1_Muons);		       
                       if(I<0)continue;
                       if(L1_Muons[I].pt()>=7) L1_Mu_RecoEff_Vs_Beta_Distrib->Fill(MC_Cand[i].p() / MC_Cand[i].energy());
           if(PassL1 && L1_Trigger_Bits_tmp[GetL1Path("L1_SingleMu7")] ){
                       HLT_Mu_RecoEff_Vs_Beta_N->Fill(MC_Cand[i].p() / MC_Cand[i].energy());
                       int I = HSCP_Trigger_ClosestHLTMuon(MC_Cand[i].phi(), MC_Cand[i].eta(),0.3,HLT_Muons);
                       if( I>=0 && HLT_Muons[I].pt()>=16) HLT_Mu_RecoEff_Vs_Beta_Distrib->Fill(MC_Cand[i].p() / MC_Cand[i].energy());
           }
        }
  }



   // RECO EFF MET Vs BETA

  if(HSCPbeta>=0 && HSCPbeta<1 && PassL1  && L1_Trigger_Bits_tmp[GetL1Path("L1_ETM40")] &&  HLT_MET.size()>0){
     reco::Particle::LorentzVector MCMET;
     for(unsigned int i=0;i<MC_Cand.size();i++){
           if(abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1){
   		MCMET += MC_Cand[i].p4();
           }
     }
     HLT_MET_RecoEff_Vs_MaxBeta_N->Fill(HSCPbeta);
     double dPhi = HSCP_Trigger_DeltaPhi(HLT_MET[0].phi(), MCMET.phi() );
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
  fprintf(f,"Number of Events with a central HSCP (eta < %6.2f)\t= %i\n",EtaOfHSCP, NEvents);
  fprintf(f,"Ratio  of Events with a central HSCP (eta < %6.2f)\t= %5.2f%%\n",EtaOfHSCP, NEvents/(0.01*NEventsBeforeEtaCut));


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
//  std::vector<unsigned int> L1Ordered = L1OrderByIncEff();
  for(unsigned int j=0;j<L1Ordered.size();j++){
      unsigned int i = L1Ordered[j];
      fprintf(f,"L1 Path %3i %30s\t: Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",i,L1_Names[i].c_str(),L1_Accepted[i]/(0.01*NEvents),L1_Rejected[i]/(0.01*NEvents),(NEvents-L1_Accepted[i]-L1_Rejected[i])/(0.01*NEvents));
  }

  fprintf(f,"Global Decision\t\t\t\t\t: Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",L1_Global_Accepted/(0.01*NEvents),L1_Global_Rejected/(0.01*NEvents),(NEvents-L1_Global_Accepted-L1_Global_Rejected)/(0.01*NEvents) );


  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ HLT TRIGGER SUMMARY @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  std::vector<unsigned int> HLTOrdered = HLTOrderByEff();
//  std::vector<unsigned int> HLTOrdered = HLTOrderByIncEff();
  for(unsigned int j=0;j<HLTOrdered.size();j++){	
	unsigned int i=HLTOrdered[j];	
	fprintf(f,"HLT Path %3i %30s\t: Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",i,HLT_Names[i].c_str(),HLT_Accepted[i]/(0.01*NEvents), HLT_Rejected[i]/(0.01*NEvents), HLT_Error[i]/(0.01*NEvents));
  }
  fprintf(f,"HLT Global Decision : NEvent = %6i\t\t  Accepted = %6.2f%%\t    Rejected = %6.2f%%\t    Errors = %6.2f%%\n",NEvents,HLT_Global_Accepted/(0.01*NEvents), HLT_Global_Rejected/(0.01*NEvents), (NEvents-HLT_Global_Accepted-HLT_Global_Rejected)/(0.01*NEvents) );


  L1Ordered  = L1OrderByIncEff();
  HLTOrdered = HLTOrderByIncEff();


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
		bool WrongOthers = false;
		for(unsigned int k=0;k<TableL1_N;k++){WrongOthers = WrongOthers || i==L1Ordered[k];}
		if(!WrongOthers)  others = others || (L1_Trigger_Bits[e])[i];
        }

        if( others ){
                TableL1_AbsEff[TableL1_N]++;    if(!Inc)TableL1_IncEff[TableL1_N]++;        Inc = true;
        }


	bool L1Global = HSCP_Trigger_L1GlobalDecision(L1_Trigger_Bits[e]);
        if(  L1Global ){	
                TableL1_AbsEff[TableL1_N+1]++;  if(!Inc)TableL1_IncEff[TableL1_N+1]++;      Inc = true;
        }



        // TABLE L1 Bis

        Inc = false;
        for(unsigned int k=0;k<TableL1Bis_N;k++){
           if( (L1_Trigger_Bits[e])[GetL1Path(TableL1Bis_Sequence[k].c_str())]){
                TableL1Bis_AbsEff[k]++;    if(!Inc)TableL1Bis_IncEff[k]++;        Inc = true;
           }
        }

        others = false;
        for(unsigned int i=0;i<L1_NPath && others == false;i++){
                bool WrongOthers = false;
                for(unsigned int k=0;k<TableL1Bis_N;k++){WrongOthers = WrongOthers || (int)i == GetL1Path( TableL1Bis_Sequence[k].c_str()) ;}
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
                bool WrongOthers = false;
                for(unsigned int k=0;k<TableHLT_N;k++){WrongOthers = WrongOthers || i==(unsigned int)HLTOrdered[k];}
                if(!WrongOthers)  others = others || (HLT_Trigger_Bits[e])[i];
        }

        if( others ){
                TableHLT_AbsEff[TableHLT_N+1]++;if(!Inc)TableHLT_IncEff[TableHLT_N+1]++;  Inc = true;
        }

        // TABLE HLT Bis

        tot = false;
        Inc = false;
        for(unsigned int k=0;k<TableHLTBis_N;k++){
           if( (HLT_Trigger_Bits[e])[GetHLTPath(TableHLTBis_Sequence[k].c_str())]){
                TableHLTBis_AbsEff[k]++;  if(!Inc)TableHLTBis_IncEff[k]++;      Inc = true;
           }
           tot = tot || (HLT_Trigger_Bits[e])[GetHLTPath(TableHLTBis_Sequence[k].c_str())];
        }

        if(tot){
                TableHLTBis_AbsEff[TableHLTBis_N]++;
        }

        others = false;
        for(unsigned int i=0;i<HLT_NPath && others == false;i++){
                bool WrongOthers = false;
                for(unsigned int k=0;k<TableHLTBis_N;k++){WrongOthers = WrongOthers || (int)i == GetHLTPath(TableHLTBis_Sequence[k].c_str()) ;}
                if(!WrongOthers)  others = others || (HLT_Trigger_Bits[e])[i];
        }

        if( others ){
                TableHLTBis_AbsEff[TableHLTBis_N+1]++;if(!Inc)TableHLTBis_IncEff[TableHLTBis_N+1]++;  Inc = true;
        }
  }

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ L1  TRIGGER TABLE  @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%40s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableL1_N;k++){
     fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n",L1_Names[L1Ordered[k]].c_str(),TableL1_AbsEff[k]/(0.01*NEvents), TableL1_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n","L1_Others"   ,TableL1_AbsEff[TableL1_N]/(0.01*NEvents), TableL1_IncEff[TableL1_N]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n","L1_Total"    ,TableL1_AbsEff[TableL1_N+1]/(0.01*NEvents), TableL1_IncEff[TableL1_N+1]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");


  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ HLT TRIGGER TABLE @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%40s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableHLT_N;k++){
      fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n",HLT_Names[HLTOrdered[k]].c_str(),TableHLT_AbsEff[k]/(0.01*NEvents), TableHLT_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n","HLT_Previous_Paths",TableHLT_AbsEff[TableHLT_N]/(0.01*NEvents), TableHLT_IncEff[TableHLT_N]/(0.01*NEvents));
  fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n","HLT_Others"        ,TableHLT_AbsEff[TableHLT_N+1]/(0.01*NEvents), TableHLT_IncEff[TableHLT_N+1]/(0.01*NEvents)
);
  fprintf(f,"----------------------------------------------------------\n");


  if(TableL1Bis_N>0){
  
  fprintf(f,"\n\n\n<><><> Personal Trigger Path Sequence <><><>\n\n\n");

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ L1  PERSONAL TRIGGER TABLE  @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%40s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableL1Bis_N;k++){
     fprintf(f,"%40s |     %6.2f%%     |     %6.2f%%\n",TableL1Bis_Sequence[k].c_str(),TableL1Bis_AbsEff[k]/(0.01*NEvents), TableL1Bis_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"%40s |     %6.1f%%     |     %6.1f%%\n","L1_Others"   ,TableL1Bis_AbsEff[TableL1Bis_N]/(0.01*NEvents), TableL1Bis_IncEff[TableL1Bis_N]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%40s |     %6.1f%%     |     %6.1f%%\n","L1_Total"    ,TableL1Bis_AbsEff[TableL1Bis_N+1]/(0.01*NEvents), TableL1Bis_IncEff[TableL1Bis_N+1]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");

  }


  if( TableHLTBis_N >0){

  fprintf(f,"\n@@@@@@@@@@@@@@@@@@@@@ HLT PERSONAL TRIGGER TABLE @@@@@@@@@@@@@@@@@@@@@@@\n\n");
  fprintf(f,"%40s | %15s | %15s\n"  ,"Trigger","Absolute Eff", "Incremental Eff");
  fprintf(f,"----------------------------------------------------------\n");
  for(unsigned int k=0;k<TableHLTBis_N;k++){
      fprintf(f,"%40s |     %6.1f%%     |     %6.1f%%\n",TableHLTBis_Sequence[k].c_str(),TableHLTBis_AbsEff[k]/(0.01*NEvents), TableHLTBis_IncEff[k]/(0.01*NEvents));}
  fprintf(f,"----------------------------------------------------------\n");
  fprintf(f,"%40s |     %6.1f%%     |     %6.1f%%\n","HLT_Previous_Paths",TableHLTBis_AbsEff[TableHLTBis_N]/(0.01*NEvents), TableHLTBis_IncEff[TableHLTBis_N]/(0.01*NEvents));
  fprintf(f,"%40s |     %6.1f%%     |     %6.1f%%\n","HLT_Others"        ,TableHLTBis_AbsEff[TableHLTBis_N+1]/(0.01*NEvents), TableHLTBis_IncEff[TableHLTBis_N+1]/(0.01*NEvents));
  fprintf(f,"----------------------------------------------------------\n");



  }
  fclose(f);
}


std::vector<unsigned int>
HSCP_Trigger::L1OrderByEff(){
  std::vector<unsigned int> To_return;
  unsigned int max = 0;  int I=-1;
  do{      
      max = 0;  I=-1;
      for(unsigned int i=0;i<L1_NPath;i++){
          bool AlreadyIn = false;
          for(unsigned int j=0;j<To_return.size();j++){AlreadyIn = AlreadyIn || (i==To_return[j]);}
          if(!AlreadyIn && L1_Accepted[i]>=max){ max = L1_Accepted[i];	I =i; }
      }
      if(I!=-1)To_return.push_back(I);
  }while(I!=-1);
  return To_return;
}


std::vector<unsigned int>
HSCP_Trigger::L1OrderByIncEff(){
	
  std::vector<unsigned int> To_return;
  unsigned int max = 0;  int I=-1;

  do{
      max = 0;  I=-1;
      for(unsigned int i=0;i<L1_NPath;i++){
          bool AlreadyIn = false;
          for(unsigned int j=0;j<To_return.size();j++){AlreadyIn = AlreadyIn || (i==To_return[j]);}

	  //compute the inc eff
	  unsigned int Inc = 0;
	  bool IncShouldBeNull = false;

	  for(unsigned int e=0;e<L1_Trigger_Bits.size();e++){
		IncShouldBeNull = false;
		for(unsigned int prev=0;prev<To_return.size();prev++){
			if((L1_Trigger_Bits[e])[ To_return[prev] ]==1 ) IncShouldBeNull=true;
		}
		if( (L1_Trigger_Bits[e])[i]==1 && !IncShouldBeNull ) Inc++;
	  }
          if(!AlreadyIn && Inc>=max){ max = Inc;  I=i; }
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
          bool AlreadyIn = false;
          for(unsigned int j=0;j<To_return.size();j++){AlreadyIn = AlreadyIn || (i==To_return[j]);}
          if(!AlreadyIn && HLT_Accepted[i]>=max){ max = HLT_Accepted[i];  I =i; }
      }
      if(I!=-1)To_return.push_back(I);
  }while(I!=-1);
  return To_return;
}

std::vector<unsigned int>
HSCP_Trigger::HLTOrderByIncEff(){

  std::vector<unsigned int> To_return;
  unsigned int max = 0;  int I=-1;

  do{
      max = 0;  I=-1;
      for(unsigned int i=0;i<HLT_NPath;i++){
          bool AlreadyIn = false;
          for(unsigned int j=0;j<To_return.size();j++){AlreadyIn = AlreadyIn || (i==To_return[j]);}

          //compute the inc eff
          unsigned int Inc = 0;
          bool IncShouldBeNull = false;

          for(unsigned int e=0;e<HLT_Trigger_Bits.size();e++){
                IncShouldBeNull = false;
                for(unsigned int prev=0;prev<To_return.size();prev++){
                        if((HLT_Trigger_Bits[e])[ To_return[prev] ]==1 ) IncShouldBeNull=true;
                }
                if( (HLT_Trigger_Bits[e])[i]==1 && !IncShouldBeNull ) Inc++;
          }
          if(!AlreadyIn && Inc>=max){ max = Inc;  I=i; }
      }
      if(I!=-1)To_return.push_back(I);
  }while(I!=-1);
  return To_return;
} 

int HSCP_Trigger::GetL1Path(const char* name)
{
   for(unsigned int i=0;i<L1_NPath;i++){
        if(strcmp(name,L1_Names[i].c_str())==0)return i;
   }
   return -1;
}

int HSCP_Trigger::GetHLTPath(const char* name)
{
   for(unsigned int i=0;i<HLT_NPath;i++){
        if(strcmp(name,HLT_Names[i].c_str())==0)return i;
   }
   return -1;
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

//define this as a plug-in
DEFINE_FWK_MODULE(HSCP_Trigger);




