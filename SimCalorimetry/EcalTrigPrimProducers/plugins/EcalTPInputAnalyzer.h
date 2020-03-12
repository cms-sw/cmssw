#ifndef EcalTPInputAnalyzer_h
#define EcalTPInputAnalyzer_h
// -*- C++ -*-
//
// Class:      EcalTPInutAnalyzer
//
/**\class EcalTPInutAnalyzer

 Description: rereads the result of the EcalTrigPrimProducer

*/
//
// Original Author:  Ursula Berthon
//         Created:  Thu Jul 4 11:38:38 CEST 2005
//
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TFile.h>
#include <TH1I.h>
#include <string>
#include <vector>

//
// class declaration
//

class EcalTPInputAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalTPInputAnalyzer(const edm::ParameterSet &);
  ~EcalTPInputAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

private:
  // for histos of nr of hits
  std::vector<std::string> ecal_parts_;
  TH1I *ecal_et_[2];
  TH1I *ecal_tt_[2];
  TH1I *ecal_fgvb_[2];
  TH1I *histEndc, *histBar;
  TFile *histfile_;

  std::string ebLabel_;
  std::string eeLabel_;
  std::string producer_;
};
#endif
