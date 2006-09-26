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
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
//#include "SimTracker/TrackAssociation/interface/TrackAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"

#include <iostream>
#include <string>
#include <map>
#include <set>

class TrackerHitAssociator;
//class TrackAssociator;
class TrackAssociatorByHits;
class TrackAssociatorBase;

class testTrackAssociator : public edm::EDAnalyzer {
  
 public:
  explicit testTrackAssociator(const edm::ParameterSet& conf);
  
  virtual ~testTrackAssociator();

  virtual void beginJob(const EventSetup&);
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  edm::ParameterSet conf_;
  bool doPixel_, doStrip_; 
  TrackAssociatorByChi2 * associator;
  //  TrackAssociatorBase * tassociator;
  TrackAssociatorByHits * tassociator;
};

#endif
