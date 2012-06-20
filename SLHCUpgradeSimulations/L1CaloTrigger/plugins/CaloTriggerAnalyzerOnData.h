//
// Original Author:  Isobel Ojalvo
// U.W. Madison
//



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
class CaloTriggerAnalyzerOnData : public edm::EDAnalyzer {
   public:
      explicit CaloTriggerAnalyzerOnData(const edm::ParameterSet&);
      ~CaloTriggerAnalyzerOnData();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      //Inputs
      edm::InputTag SLHCsrc_;
      edm::InputTag LHCsrc_;
      edm::InputTag LHCisosrc_;
      //edm::InputTag ref_;
      double iso_;
      double DR_;
      double threshold_;
      double maxEta_;

      float highestGenPt;
      float secondGenPt;
      float highPt;
      float secondPtf;

      TH1F * SLHCpt;
      TH1F * LHCpt;

      TH1F * eta;

      TH1F * ptNum;   
      TH1F * ptDenom;
      TH1F * etaNum;
      TH1F * etaDenom;
      TH1F * pt;
      TH1F * dPt;
      TH1F * dEta;
      TH1F * dPhi;

      TH1F * LHChighestPt;
      TH1F * LHCsecondPt;
      TH1F * SLHChighestPt;
      TH1F * SLHCsecondPt;

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





