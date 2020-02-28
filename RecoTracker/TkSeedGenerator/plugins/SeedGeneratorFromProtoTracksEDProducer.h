#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class dso_hidden SeedGeneratorFromProtoTracksEDProducer : public edm::stream::EDProducer<> {
public:
  SeedGeneratorFromProtoTracksEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromProtoTracksEDProducer() override {}
  void produce(edm::Event& ev, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ParameterSet theConfig;
  const double originHalfLength;
  const double originRadius;
  const bool useProtoTrackKinematics;
  const bool useEventsWithNoVertex;
  const std::string builderName;
  const bool usePV_;
  const bool includeFourthHit_;
  const edm::EDGetTokenT<reco::TrackCollection> theInputCollectionTag;
  const edm::EDGetTokenT<reco::VertexCollection> theInputVertexCollectionTag;
};
#endif
