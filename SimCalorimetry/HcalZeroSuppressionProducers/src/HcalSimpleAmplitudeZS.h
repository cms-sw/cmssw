#ifndef HCALSIMPLEAMPLITUDEZS_H
#define HCALSIMPLEAMPLITUDEZS_H 1

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "HcalZSAlgoEnergy.h"

#include <string>

/** \class HcalSimpleAmplitudeZS
	
\author J. Mans - Minnesota
*/
class HcalSimpleAmplitudeZS : public edm::stream::EDProducer<> {
public:
  explicit HcalSimpleAmplitudeZS(const edm::ParameterSet& ps);
  virtual ~HcalSimpleAmplitudeZS();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  std::unique_ptr<HcalZSAlgoEnergy> hbhe_,ho_,hf_,hfQIE10_,hbheQIE11_;
  std::string inputLabel_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_hfQIE10_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_hbheQIE11_;
};

#endif
