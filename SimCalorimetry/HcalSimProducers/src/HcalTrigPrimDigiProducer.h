#ifndef HcalTrigPrimDigiProducer_h
#define HcalTrigPrimDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalTrigPrimDigiProducer : public edm::EDProducer
{
public:

  explicit HcalTrigPrimDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalTrigPrimDigiProducer() {}

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

  HcalCoderFactory theCoderFactory;
  HcalTriggerPrimitiveAlgo theAlgo;
  edm::InputTag inputLabel_;
  int hcalScale_, emScale_, jetScale_;
};

#endif

