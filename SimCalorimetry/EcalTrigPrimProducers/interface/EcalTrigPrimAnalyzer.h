// -*- C++ -*-
//
// Class:      EcalTrigPrimAnalyzer
// 
/**\class EcalTrigPrimAnalyzer

 Description: rereads the result of the EcalTrigPrimProducer

*/
//
// Original Author:  Ursula Berthon
//         Created:  Thu Jul 4 11:38:38 CEST 2005
// $Id: EcalTrigPrimAnalyzer.h,v 1.1 2006/07/04 16:32:04 uberthon Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>
#include <TH1I.h>
#include <TFile.h>


//
// class declaration
//

class EcalTrigPrimAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EcalTrigPrimAnalyzer(const edm::ParameterSet&);
      ~EcalTrigPrimAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:

  // for histos of nr of hits
  std::vector<std::string> ecal_parts_;
  TH1I * ecal_et_[2];
  TH1I * ecal_tt_[2];
  TH1I * ecal_fgvb_[2];

  TFile *histfile_;

};

