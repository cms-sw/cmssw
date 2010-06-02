#ifndef ESZEROSUPPRESSIONPRODUCER_H
#define ESZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "CondFormats/ESObjects/interface/ESThresholds.h"
#include "CondFormats/DataRecord/interface/ESThresholdsRcd.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"

class ESZeroSuppressionProducer : public edm::EDProducer
{
 public:
    
  explicit ESZeroSuppressionProducer(const edm::ParameterSet& ps);
  virtual ~ESZeroSuppressionProducer();
  
  /**Produces the EDM products,*/
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);
  
  
 private:

  std::string digiProducer_;
  std::string ESdigiCollection_;
  std::string ESZSdigiCollection_;

  edm::ESHandle<ESThresholds> esthresholds_;
  edm::ESHandle<ESPedestals> espeds_;
};

#endif 
