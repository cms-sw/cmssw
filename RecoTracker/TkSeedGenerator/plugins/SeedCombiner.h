#ifndef RecoTracker_TkSeedGenerator_SeedCombiner_H
#define RecoTracker_TkSeedGenerator_SeedCombiner_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Event; class EventSetup; }


class SeedCombiner : public edm::EDProducer {
public:

  SeedCombiner(const edm::ParameterSet& cfg);
  ~SeedCombiner();

  virtual void beginJob(const edm::EventSetup& es);

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:
  //FIXME:
  //This part may be generalized in order to accept and merge more than 2 collections of seeds
  std::string SeedPairCollectionName_; //used to select what tracks to read from configuration file
  std::string SeedTripletCollectionName_; //used to select what tracks to read from configuration file
};

#endif
