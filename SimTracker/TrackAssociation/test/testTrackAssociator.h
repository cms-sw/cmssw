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

namespace reco {
  class TrackToTrackingParticleAssociator;
}

class testTrackAssociator : public edm::EDAnalyzer {
  
 public:
  testTrackAssociator(const edm::ParameterSet& conf);
  virtual ~testTrackAssociator();
  virtual void beginJob() {}  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  reco::TrackToTrackingParticleAssociator const * associatorByChi2;
  reco::TrackToTrackingParticleAssociator const * associatorByHits;
  edm::InputTag tracksTag, tpTag, simtracksTag, simvtxTag;
};

#endif
