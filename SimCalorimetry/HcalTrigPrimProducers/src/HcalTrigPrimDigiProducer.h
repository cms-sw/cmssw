#ifndef HcalTrigPrimDigiProducer_h
#define HcalTrigPrimDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

class HcalTrigPrimDigiProducer : public edm::EDProducer
{
public:

  explicit HcalTrigPrimDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalTrigPrimDigiProducer() {}

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  HcalTriggerPrimitiveAlgo theAlgo;
  std::vector<edm::InputTag> inputLabel_;
  bool runFrontEndFormatError;
  bool runZS;
};

#endif

