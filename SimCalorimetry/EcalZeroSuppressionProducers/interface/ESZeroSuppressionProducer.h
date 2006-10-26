#ifndef ESZEROSUPPRESSIONPRODUCER_H
#define ESZEROSUPPRESSIONPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"

class ESZeroSuppressionProducer : public edm::EDProducer
{
 public:
    
  explicit ESZeroSuppressionProducer(const edm::ParameterSet& ps);
  virtual ~ESZeroSuppressionProducer();
  
  /**Produces the EDM products,*/
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);
  
  
 private:

  void checkGeometry(const edm::EventSetup & eventSetup);
  void updateGeometry();
 
  const CaloGeometry * theGeometry;

  std::string digiProducer_;
  std::string ESdigiCollection_;
  std::string ESZSdigiCollection_;

  ESRecHitSimAlgo *algo_;

  int ESGain;
  int ESBaseline;
  double ESMIPADC;
  double ESMIPkeV;
  double ESNoiseSigma;
  double ESThreshold;

};

#endif 
