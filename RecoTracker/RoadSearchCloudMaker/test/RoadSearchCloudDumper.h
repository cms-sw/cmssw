#ifndef RecoTrackerRoadSearchCloudMakerRoadSearchCloudDumper_h
#define RecoTrackerRoadSearchCloudMakerRoadSearchCloudDumper_h

//
// Package:         RecoTracker/RoadSearchCloudMaker/test
// Class:           RoadSearchCloudDumper.cc
// 
// Description:     Cloud Dumper
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Mon Feb  5 21:24:36 UTC 2007
//
// $Author: wmtan $
// $Date: 2010/02/11 00:14:43 $
// $Revision: 1.4 $
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RoadSearchCloudDumper : public edm::EDAnalyzer {
 public:
  RoadSearchCloudDumper(const edm::ParameterSet& conf);
  ~RoadSearchCloudDumper();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 private:
  edm::InputTag roadSearchCloudsInputTag_;

  std::string ringsLabel_;
};

#endif
