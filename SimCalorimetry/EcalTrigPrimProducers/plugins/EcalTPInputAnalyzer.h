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
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <TH1I.h>
#include <string>
#include <vector>

//
// class declaration
//

class EcalTPInputAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EcalTPInputAnalyzer(const edm::ParameterSet &);
  ~EcalTPInputAnalyzer() override = default;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  // for histos of nr of hits
  std::vector<std::string> ecal_parts_;
  TH1I *ecal_et_[2];
  TH1I *ecal_tt_[2];
  TH1I *ecal_fgvb_[2];
  TH1I *histEndc, *histBar;

  const std::string producer_;
  const std::string ebLabel_;
  const std::string eeLabel_;
  const edm::EDGetTokenT<EBDigiCollection> ebToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeToken_;
};
#endif
