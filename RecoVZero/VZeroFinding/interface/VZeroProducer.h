#ifndef _VZeroProducer_h_
#define _VZeroProducer_h_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class VZeroProducer :  public edm::EDProducer {

public:
  explicit VZeroProducer(const edm::ParameterSet& pset);

  ~VZeroProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  edm::ParameterSet pset_;

  float minImpactPositiveDaughter,
        minImpactNegativeDaughter;
};
#endif

