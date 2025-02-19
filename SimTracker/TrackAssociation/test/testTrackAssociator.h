#ifndef testTrackAssociator_h
#define testTrackAssociator_h

#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>
#include <map>
#include <set>

class TrackAssociatorBase;

class testTrackAssociator : public edm::EDAnalyzer {
  
 public:
  testTrackAssociator(const edm::ParameterSet& conf);
  virtual ~testTrackAssociator();
  virtual void beginJob() {}  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  TrackAssociatorBase * associatorByChi2;
  TrackAssociatorBase * associatorByHits;
  edm::InputTag tracksTag, tpTag, simtracksTag, simvtxTag;
};

#endif
