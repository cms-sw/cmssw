#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits;
class TrackingRegionProducer;
class QuadrupletSeedMerger;

class dso_hidden SeedGeneratorFromRegionHitsEDProducer : public edm::stream::EDProducer<> {
public:

  SeedGeneratorFromRegionHitsEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromRegionHitsEDProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  std::unique_ptr<SeedGeneratorFromRegionHits> theGenerator;
  std::unique_ptr<TrackingRegionProducer> theRegionProducer;
  ClusterChecker theClusterCheck;
  std::unique_ptr<QuadrupletSeedMerger> theMerger_;

  std::string moduleName;

  bool theSilentOnClusterCheck;
};

#endif
