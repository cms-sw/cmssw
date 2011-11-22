#ifndef SIMCALORIMETRY_HCALTRIGPRIMPRODUCERS_SRC_HCALUPGRADETRIGPRIMDIGIPRODUCER_H
#define SIMCALORIMETRY_HCALTRIGPRIMPRODUCERS_SRC_HCALUPGRADETRIGPRIMDIGIPRODUCER_H

//------------------------------------------------------
// Include files
//------------------------------------------------------

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Algorithm
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalUpgradeTriggerPrimitiveAlgo.h"

class HcalUpgradeTrigPrimDigiProducer : public edm::EDProducer {
public:
  explicit HcalUpgradeTrigPrimDigiProducer(const edm::ParameterSet&);
  ~HcalUpgradeTrigPrimDigiProducer();
  
private:

  //------------------------------------------------------
  // Analaysis functions
  //------------------------------------------------------

  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;  
  
  //------------------------------------------------------
  // InputTags
  //------------------------------------------------------
  
  const edm::InputTag m_hbheDigisTag;
  const edm::InputTag m_hfDigisTag  ;

  //------------------------------------------------------
  // Algorithm
  //------------------------------------------------------
 
  HcalUpgradeTriggerPrimitiveAlgo * m_hcalUpgradeTriggerPrimitiveDigiAlgo;
  
};

#endif
