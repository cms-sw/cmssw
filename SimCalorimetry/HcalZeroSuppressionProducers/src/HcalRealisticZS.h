#ifndef HCALSIMPLEREALISTICZS_H
#define HCALSIMPLEREALISTICZS_H 1

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "HcalZSAlgoRealistic.h"

#include <string>

/** \class HcalSimpleRealisticZS

\author J. Mans - Minnesota
*/
class HcalRealisticZS : public edm::stream::EDProducer<> {
public:
  explicit HcalRealisticZS(const edm::ParameterSet &ps);
  ~HcalRealisticZS() override;
  void produce(edm::Event &e, const edm::EventSetup &c) override;

private:
  std::unique_ptr<HcalZSAlgoRealistic> algo_;
  std::string inputLabel_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_hfQIE10_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_hbheQIE11_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbService_;
};

#endif
