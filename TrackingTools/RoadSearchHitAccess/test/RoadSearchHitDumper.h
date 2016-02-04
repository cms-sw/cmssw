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
// $Author: wmtan $
// $Date: 2010/02/11 00:15:16 $
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

  std::string ringsLabel_;
};

#endif
