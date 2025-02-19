#ifndef RECOTRACKER_RINGTEST_H
#define RECOTRACKER_RINGTEST_H

//
// Package:         RecoTracker/RingMakerESProducer/test
// Class:           RingTest
// 
// Description:     test rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Fri Dec  8 10:15:02 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/01 07:45:12 $
// $Revision: 1.2 $
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RingTest : public edm::EDAnalyzer {
 public:
  explicit RingTest( const edm::ParameterSet& );
  ~RingTest();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
 private:
  // ----------member data ---------------------------
  bool dumpRings_;
  std::string fileName_;
  std::string ringLabel_;

};

#endif
