#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracks_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracks_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm { class ParameterSet; class Event; class EventSetup; }; 


class SeedGeneratorFromProtoTracks : public edm::EDProducer {
public:
  SeedGeneratorFromProtoTracks(const edm::ParameterSet& cfg);
  virtual ~SeedGeneratorFromProtoTracks(){}
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
private:
  edm::InputTag theInputCollectionTag;  
};
#endif
