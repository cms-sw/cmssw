//
// Original Author:  Michail BACHTIS
//         Created:  Tue Jul 22 12:21:36 CEST 2008
// $Id: CaloTriggerAnalyzer.h,v 1.4 2010/09/28 19:11:29 iross Exp $



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

#include "TH1F.h"
#include "TProfile.h"
class CaloTriggerAnalyzer : public edm::EDAnalyzer {
   public:
      explicit CaloTriggerAnalyzer(const edm::ParameterSet&);
      ~CaloTriggerAnalyzer();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      //Inputs
      edm::InputTag src_;
      edm::InputTag ref_;
      double DR_;
      double threshold_;
      double maxEta_;

      float highestGenPt;
      float secondGenPt;
      float highPt;
      float secondPtf;
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
      TH1F * dR;
      TProfile * RPtEta;
      TProfile * RPtEtaFull;
};





