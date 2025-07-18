// -*- C++ -*-
//
// Package:    ExtraFromSeeds
// Class:      ExtraFromSeeds
//
/**\class ExtraFromSeeds ExtraFromSeeds.cc RecoTracker/ExtraFromSeeds/src/ExtraFromSeeds.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Fri Feb 17 12:03:11 CET 2012
//
//

// system include files
#include <memory>

// user include files
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class ExtraFromSeeds : public edm::global::EDProducer<> {
public:
  explicit ExtraFromSeeds(const edm::ParameterSet&);
  ~ExtraFromSeeds() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<reco::TrackCollection> tracks_;
  typedef std::vector<unsigned int> ExtremeLight;

  // ----------member data ---------------------------
};

//
// constructors and destructor
//
ExtraFromSeeds::ExtraFromSeeds(const edm::ParameterSet& iConfig)
    : tracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))) {
  produces<ExtremeLight>();
  produces<TrackingRecHitCollection>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void ExtraFromSeeds::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // in
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracks_, tracks);

  // out
  std::unique_ptr<ExtremeLight> exxtralOut(new ExtremeLight());
  exxtralOut->resize(tracks->size());

  std::unique_ptr<TrackingRecHitCollection> hitOut(new TrackingRecHitCollection());
  TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
  hitOut->reserve(3 * tracks->size());

  for (unsigned int ie = 0; ie != tracks->size(); ++ie) {
    const reco::Track& track = (*tracks)[ie];
    const reco::TrackExtra& extra = *track.extra();
    //only for high purity tracks
    if (!track.quality(reco::TrackBase::highPurity))
      continue;

    (*exxtralOut)[ie] = extra.seedRef()->nHits();
    for (auto const& seedHit : extra.seedRef()->recHits()) {
      hitOut->push_back(seedHit.clone());
    }
  }

  iEvent.put(std::move(exxtralOut));
  iEvent.put(std::move(hitOut));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ExtraFromSeeds::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ExtraFromSeeds);
