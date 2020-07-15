#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHitsEDProducer_H
#include "FWCore/Utilities/interface/Visibility.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class SeedGeneratorFromRegionHits;
class TrackingRegionProducer;

class dso_hidden SeedGeneratorFromRegionHitsEDProducer : public edm::stream::EDProducer<> {
public:
  SeedGeneratorFromRegionHitsEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromRegionHitsEDProducer() override;

  void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  std::unique_ptr<SeedGeneratorFromRegionHits> theGenerator;
  std::unique_ptr<TrackingRegionProducer> theRegionProducer;
  ClusterChecker theClusterCheck;

  std::string moduleName;

  bool theSilentOnClusterCheck;
};

#endif
