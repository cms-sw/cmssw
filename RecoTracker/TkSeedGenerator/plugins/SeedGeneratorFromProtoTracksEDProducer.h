#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H
#include "FWCore/Utilities/interface/Visibility.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SeedFromConsecutiveHitsCreator.h"

#include <string>

class dso_hidden SeedGeneratorFromProtoTracksEDProducer : public edm::stream::EDProducer<> {
public:
  SeedGeneratorFromProtoTracksEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromProtoTracksEDProducer() override {}
  void produce(edm::Event& ev, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const double originHalfLength;
  const double originRadius;
  const bool useProtoTrackKinematics;
  const bool useEventsWithNoVertex;
  const bool usePV_;
  const bool includeFourthHit_;
  const edm::EDGetTokenT<reco::TrackCollection> theInputCollectionTag;
  const edm::EDGetTokenT<reco::VertexCollection> theInputVertexCollectionTag;
  SeedFromConsecutiveHitsCreator seedCreator_;
};
#endif
