#ifndef HcalDigiProducer_h
#define HcalDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"


class HcalDigiProducer : public edm::EDProducer
{
public:

  explicit HcalDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  virtual void beginRun(edm::Run&, edm::EventSetup const&);
  virtual void endRun(edm::Run&, edm::EventSetup const&);


private:
  HcalDigitizer theDigitizer;
};

#endif

