#ifndef RECOTRACKER_ROADMAPTEST_H
#define RECOTRACKER_ROADMAPTEST_H

//
// Package:         RecoTracker/RoadMapRecord
// Class:           RoadMapTest
// 
// Description:     test roads
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sun Feb  4 19:15:56 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/02/05 19:22:46 $
// $Revision: 1.1 $
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RoadMapTest : public edm::EDAnalyzer {
 public:
  explicit RoadMapTest( const edm::ParameterSet& );
  ~RoadMapTest();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  bool dumpRoads_;
  std::string fileName_;
  std::string roadLabel_;  
};

#endif
