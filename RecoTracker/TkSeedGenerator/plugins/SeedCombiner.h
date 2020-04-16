#ifndef RecoTracker_TkSeedGenerator_SeedCombiner_H
#define RecoTracker_TkSeedGenerator_SeedCombiner_H

#include <vector>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Visibility.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm

class dso_hidden SeedCombiner : public edm::stream::EDProducer<> {
public:
  SeedCombiner(const edm::ParameterSet& cfg);
  ~SeedCombiner() override;

  void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  std::vector<edm::EDGetTokenT<TrajectorySeedCollection>> inputCollections_;
  bool reKeing_;
  std::vector<edm::InputTag> clusterRemovalInfos_;
  std::vector<edm::EDGetTokenT<reco::ClusterRemovalInfo>> clusterRemovalTokens_;
};

#endif
