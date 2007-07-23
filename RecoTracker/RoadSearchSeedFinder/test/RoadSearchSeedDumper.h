#ifndef RecoTrackerRoadSearchSeedFinderRoadSearchSeedDumper_h
#define RecoTrackerRoadSearchSeedFinderRoadSearchSeedDumper_h

//
// Package:         RecoTracker/RoadSearchSeedFinder/test
// Class:           RoadSearchSeedDumper.cc
// 
// Description:     Seed Dumper
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Mon Feb  5 21:24:36 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/03/01 08:16:18 $
// $Revision: 1.2 $
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RoadSearchSeedDumper : public edm::EDAnalyzer {
 public:
  RoadSearchSeedDumper(const edm::ParameterSet& conf);
  ~RoadSearchSeedDumper();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 private:
  edm::InputTag roadSearchSeedsInputTag_;
  std::string ringsLabel_;
};

#endif
