#ifndef RecoTracker_TkSeedGenerator_SeedCombiner_H
#define RecoTracker_TkSeedGenerator_SeedCombiner_H

#include <vector>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm { class Event; class EventSetup; class ParameterSet; }


class SeedCombiner : public edm::EDProducer {
public:

  SeedCombiner(const edm::ParameterSet& cfg);
  ~SeedCombiner();

  virtual void beginRun(edm::Run & run, const edm::EventSetup& es);

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:
  std::vector<edm::InputTag> inputCollections_;
};

#endif
