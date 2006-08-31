#ifndef testTrackAssociator_h
#define testTrackAssociator_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
//add simhit info
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociation.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"

#include <iostream>
#include <string>
#include <map>
#include <set>

class TrackerHitAssociator;
class TrackAssociator;

class testTrackAssociator : public edm::EDAnalyzer {
  
 public:
  explicit testTrackAssociator(const edm::ParameterSet& conf);
  
  virtual ~testTrackAssociator();
  
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  
 private:
  edm::ParameterSet conf_;
  bool doPixel_, doStrip_;
};

#endif
