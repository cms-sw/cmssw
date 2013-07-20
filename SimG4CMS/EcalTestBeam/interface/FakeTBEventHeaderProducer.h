#ifndef FakeTBEventHeaderProducer_H
#define FakeTBEventHeaderProducer_H
/*
 * \file FakeTBEventHeaderProducer.h
 *
 * $Id: FakeTBEventHeaderProducer.h,v 1.5 2011/05/20 17:17:34 wmtan Exp $
 *
 * Mimic the event header information
 * for the test beam simulation 
 *
 */


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"

class FakeTBEventHeaderProducer: public edm::EDProducer{

  
 public:
  
  /// Constructor
  FakeTBEventHeaderProducer(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~FakeTBEventHeaderProducer();
  
  /// Produce digis out of raw data
  void produce(edm::Event & event, const edm::EventSetup& eventSetup);
  
  // BeginJob
  //void beginJob();
  
  // EndJob
  //void endJob(void);
  

private:

  std::string ecalTBInfoLabel_;

};

#endif
