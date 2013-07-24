//
// Original Author:  Michail BACHTIS
//         Created:  Tue Jul 22 12:21:36 CEST 2008
// $Id: CaloTriggerAnalyzer2.h,v 1.1 2010/09/28 19:11:29 iross Exp $



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TH1F.h"
#include "TProfile.h"
#include "TTree.h"

class CaloTriggerAnalyzer2 : public edm::EDAnalyzer {
   public:
      explicit CaloTriggerAnalyzer2(const edm::ParameterSet&);
      ~CaloTriggerAnalyzer2();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void clearVectors();
      virtual void fillSLHCBranches(const reco::Candidate * L1object, const reco::Candidate * genParticle);
      virtual void fillLHCBranches(const reco::Candidate * L1object, const reco::Candidate * genParticle);
      
      //Inputs
      edm::InputTag ref_;
      edm::InputTag SLHCsrc_;
      edm::InputTag LHCsrc_;
      double DR_;
      double threshold_;
      double maxEta_;
      
      TTree * EventTree;
      TTree * CandTree;

      std::vector<double> gPt;
      std::vector<double> gEta;
      std::vector<double> gPhi;

      std::vector<double> SLHCL1Pt;
      std::vector<double> SLHCL1Eta;
      std::vector<double> SLHCL1Phi;
      std::vector<double> SLHCdR;
      std::vector<double> SLHCdPt;
      std::vector<double> SLHCdEta;
      std::vector<double> SLHCdPhi;
      std::vector<double> SLHCRPt;
      std::vector<double> SLHCL1Ptc;
      std::vector<double> SLHCL1Etac;
      std::vector<double> SLHCL1Phic;
      std::vector<double> SLHCdRc;
      std::vector<double> SLHCdPtc;
      std::vector<double> SLHCdEtac;
      std::vector<double> SLHCdPhic;
      std::vector<double> SLHCRPtc;
      std::vector<bool> SLHCpassSingleThresh;
      std::vector<bool> SLHCpassDoubleThresh;

      std::vector<double> LHCL1Pt;
      std::vector<double> LHCL1Eta;
      std::vector<double> LHCL1Phi;
      std::vector<double> LHCdR;
      std::vector<double> LHCdPt;
      std::vector<double> LHCdEta;
      std::vector<double> LHCdPhi;
      std::vector<double> LHCRPt;
      std::vector<double> LHCL1Ptc;
      std::vector<double> LHCL1Etac;
      std::vector<double> LHCL1Phic;
      std::vector<double> LHCdRc;
      std::vector<double> LHCdPtc;
      std::vector<double> LHCdEtac;
      std::vector<double> LHCdPhic;
      std::vector<double> LHCRPtc;
      std::vector<bool> LHCpassSingleThresh;
      std::vector<bool> LHCpassDoubleThresh;

      float ghighPt;
      float gsecondPt;
      float ghighMPt; //highest matched GEN Pt
      float SLHChighPt;
      float SLHCsecondPt;
      float LHChighPt;
      float LHCsecondPt;

      int nEvents;

      bool SLHCsingleTrigger;
      bool SLHCdoubleTrigger;
      bool LHCsingleTrigger;
      bool LHCdoubleTrigger;

      float highestGenPt;
      float secondGenPt;
      float highPt;
      float secondPtf;
      
      const reco::Candidate * L1object;
      const reco::Candidate * L1closest;

      TH1F * eta;

      TH1F * ptNum;   
      TH1F * ptDenom;
      TH1F * etaNum;
      TH1F * etaDenom;
      TH1F * pt;
      TH1F * dPt;
      TH1F * dEta;
      TH1F * dPhi;
      TH1F * highestPt;
      TH1F * secondPt;
      TH1F * highestPtGen;
      TH1F * secondPtGen;
      TH1F * RPt;
      TH1F * absEta;
      TH1F * SLHCalldR;
      TH1F * LHCalldR;
      TProfile * RPtEta;
      TProfile * RPtEtaFull;
};





