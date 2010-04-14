// -*- C++ -*-
//
// Package:    HSCP
// Class:      HSCPValidator
// 
/**\class HSCPValidator HSCPValidator.cc HSCPValidation/HSCPValidator/src/HSCPValidator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Seth Cooper,27 1-024,+41227672342,
//         Created:  Wed Apr 14 14:27:52 CEST 2010
// $Id$
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class declaration
//

class HSCPValidator : public edm::EDAnalyzer {
   public:
      explicit HSCPValidator(const edm::ParameterSet&);
      ~HSCPValidator();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::string intToString(int num);

      // ----------member data ---------------------------

      // GEN section
      edm::InputTag label_;
      std::vector<int> particleIds_;
      int particleStatus_;

      std::map<int,int> particleIdsFoundMap_;

      TH1F* particleEtaHist_;
      TH1F* particlePhiHist_;
      TH1F* particlePHist_;
      TH1F* particlePtHist_;
      TH1F* particleMassHist_;
      TH1F* particleStatusHist_;
      TH1F* particleBetaHist_;
      TH1F* particleBetaInverseHist_;
};
