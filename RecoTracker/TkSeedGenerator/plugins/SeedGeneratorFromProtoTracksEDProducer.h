#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Event; class EventSetup; }


class SeedGeneratorFromProtoTracksEDProducer : public edm::EDProducer {
public:
  SeedGeneratorFromProtoTracksEDProducer(const edm::ParameterSet& cfg);
  virtual ~SeedGeneratorFromProtoTracksEDProducer(){}
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;
private:
  edm::ParameterSet theConfig;
  edm::InputTag theInputCollectionTag;
  edm::InputTag theInputVertexCollectionTag;
  double originHalfLength;
  double originRadius;
  bool useProtoTrackKinematics;
  bool useEventsWithNoVertex;
  std::string builderName;
};
#endif
