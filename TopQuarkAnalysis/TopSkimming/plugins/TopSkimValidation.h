// -*- C++ -*-
//
// Package:    TopSkimValidation
// Class:      TopSkimValidation
// 
/**\class TopSkimValidation TopSkimValidation.cc TopQuarkAnalysis/TopSkimValidation/src/TopSkimValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Puneeth Kalavase
//         Created:  Thu Oct  1 17:22:48 PDT 2009
// $Id$
//
//


// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TH1F.h"
#include "TFile.h"



//
// class decleration
//

class TopSkimValidation : public edm::EDAnalyzer {
   public:
  void fillTriggerObjectInfo(unsigned int triggerIndex, std::vector<math::XYZTLorentzVectorD>& p4V, double ptcut) const; 
      explicit TopSkimValidation(const edm::ParameterSet&);
      ~TopSkimValidation();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  
  edm::Handle<edm::TriggerResults> triggerResultsH_;
  edm::Handle<trigger::TriggerEvent> triggerEventH_;
  HLTConfigProvider hltConfig_;
  unsigned int numPt; //number of reco leptons with pt > X
  unsigned int numPtHLTMatched; // number of reco leptons with pt > X matched to a HLT object 
  
  //
  std::string processName_;
  std::vector<std::string> triggerNames_;
  std::string leptonFlavor_; //electron or Muon? Should be obvious from the triggers, but still
  double leptonPtcut_; //pt cut on the leptons
  double HLTPtcut_;    //pt cut on the HLT object
  edm::InputTag leptonInputTag_; //input tag of the collection
  
  // ----------member data ---------------------------
};
