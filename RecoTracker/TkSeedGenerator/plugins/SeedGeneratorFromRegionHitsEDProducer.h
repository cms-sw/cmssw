#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits;
class TrackingRegionProducer;

class SeedGeneratorFromRegionHitsEDProducer : public edm::EDProducer {
public:

  SeedGeneratorFromRegionHitsEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromRegionHitsEDProducer();

  virtual void beginRun(edm::Run &run, const edm::EventSetup& es);
  virtual void endRun(edm::Run &run, const edm::EventSetup& es);

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:
  edm::ParameterSet theConfig;
  SeedGeneratorFromRegionHits * theGenerator; 
  TrackingRegionProducer* theRegionProducer;
  ClusterChecker theClusterCheck;
  bool theSilentOnClusterCheck;
};

#endif
