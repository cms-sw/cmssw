#ifndef FakeTBHodoscopeRawInfoProducer_H
#define FakeTBHodoscopeRawInfoProducer_H
/*
 * \file FakeTBHodoscopeRawInfoProducer.h
 *
 * $Id: FakeTBHodoscopeRawInfoProducer.h,v 1.1 2006/05/31 09:31:57 fabiocos Exp $
 *
 * Mimic the hodoscope raw information using 
 * the generated vertex of the test beam simulation 
 *
 */


#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"

class FakeTBHodoscopeRawInfoProducer: public edm::EDProducer{

  
 public:
  
  /// Constructor
  FakeTBHodoscopeRawInfoProducer(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~FakeTBHodoscopeRawInfoProducer();
  
  /// Produce digis out of raw data
  void produce(edm::Event & event, const edm::EventSetup& eventSetup);
  
  // BeginJob
  //void beginJob(const edm::EventSetup& c);
  
  // EndJob
  //void endJob(void);
  

private:

  EcalTBHodoscopeGeometry * theTBHodoGeom_;

  std::string ecalTBInfoLabel_;

};

#endif
