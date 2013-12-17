#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromProtoTracksEDProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace edm { class Event; class EventSetup; }


class SeedGeneratorFromProtoTracksEDProducer : public edm::EDProducer {
public:
  SeedGeneratorFromProtoTracksEDProducer(const edm::ParameterSet& cfg);
  virtual ~SeedGeneratorFromProtoTracksEDProducer(){}
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::ParameterSet theConfig;
  edm::EDGetTokenT<reco::TrackCollection> theInputCollectionTag;
  edm::EDGetTokenT<reco::VertexCollection> theInputVertexCollectionTag;
  double originHalfLength;
  double originRadius;
  bool useProtoTrackKinematics;
  bool useEventsWithNoVertex;
  std::string builderName;
  bool usePV_;

};
#endif
