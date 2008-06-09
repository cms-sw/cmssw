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
// $Id: EcalTrigPrimAnalyzer.h,v 1.4 2007/09/25 15:56:25 uberthon Exp $
//
//


// system include files
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>
#include <TH1I.h>
#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TH1F.h>

//
// class declaration
//

class EcalTrigPrimAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EcalTrigPrimAnalyzer(const edm::ParameterSet&);
      ~EcalTrigPrimAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
   private:

  // for histos of nr of hits
  std::vector<std::string> ecal_parts_;
  TH1I * ecal_et_[2];
  TH1I * ecal_tt_[2];
  TH1I * ecal_fgvb_[2];
  TH1I *histEndc,*histBar;
  TFile *histfile_;
  TH2F *hTPvsRechit_;
  TH1F *hTPoverRechit_;
  TTree *tree_ ;

  int iphi_, ieta_ , tpgADC_, ttf_, fg_ ;
  float eRec_, tpgGeV_ ;

  std::string label_;
  std::string producer_;
  std::string rechits_labelEB_;
  std::string  rechits_labelEE_;
  std::string  rechits_producer_;

  bool recHits_;
};

