#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits;
class TrackingRegionProducer;
class QuadrupletSeedMerger;

class SeedGeneratorFromRegionHitsEDProducer : public edm::EDProducer {
public:

  SeedGeneratorFromRegionHitsEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromRegionHitsEDProducer();

  virtual void beginRun(edm::Run const&run, const edm::EventSetup& es) override;
  virtual void endRun(edm::Run const&run, const edm::EventSetup& es) override;

  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  edm::ParameterSet theConfig;
  std::unique_ptr<SeedGeneratorFromRegionHits> theGenerator;
  TrackingRegionProducer* theRegionProducer;
  ClusterChecker theClusterCheck;
  QuadrupletSeedMerger *theMerger_;

  std::string moduleName;

  bool theSilentOnClusterCheck;
};

#endif
