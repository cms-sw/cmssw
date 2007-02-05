#ifndef TrackingToolsRoadSearchHitAccessRoadSearchHitDumper_h
#define TrackingToolsRoadSearchHitAccessRoadSearchHitDumper_h

//
// Package:         TrackingTools/RoadSearchHitAccess/test
// Class:           RoadSearchHitDumper.cc
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:26:15 $
// $Revision: 1.3 $
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RoadSearchHitDumper : public edm::EDAnalyzer {
 public:
  RoadSearchHitDumper(const edm::ParameterSet& conf);
  ~RoadSearchHitDumper();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);

 private:
  edm::InputTag matchedStripRecHitsInputTag_;
  edm::InputTag rphiStripRecHitsInputTag_;
  edm::InputTag stereoStripRecHitsInputTag_;
  edm::InputTag pixelRecHitsInputTag_;
};

#endif
