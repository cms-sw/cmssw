#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm { class ParameterSet; class Event; class EventSetup; }


class SeedGeneratorFromProtoTracksEDProducer : public edm::EDProducer {
public:
  SeedGeneratorFromProtoTracksEDProducer(const edm::ParameterSet& cfg);
  virtual ~SeedGeneratorFromProtoTracksEDProducer(){}
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
private:
  edm::InputTag theInputCollectionTag;  
};
#endif
