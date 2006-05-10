#ifndef EcaltrigprimProducer_h
#define EcaltrigprimProducer_h
  
//
// Package:        Simcalorimetry/EcalTrigPrimProducer
// Class:          EcaTtrigPrimProducer
// 
// Description:     Calls algorithms tio create EcalTriggerPrimitiveDigi-s
//                  depending on configuration, 'simple' or 'functional' algorthm is used
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class EcalTrigPrimFunctionalAlgo;
 
class EcalTrigPrimProducer : public edm::EDProducer
{
 public:
  
  explicit EcalTrigPrimProducer(const edm::ParameterSet& conf);
  
  virtual ~EcalTrigPrimProducer();
  
  void beginJob(edm::EventSetup const& setup);

  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
  EcalTrigPrimFunctionalAlgo *algo_;
 
};
  
#endif
 


