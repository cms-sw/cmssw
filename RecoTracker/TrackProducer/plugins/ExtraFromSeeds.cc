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

#include "RecoTracker/TrackProducer/plugins/ExtraFromSeeds.h"

//
// constructors and destructor
//
ExtraFromSeeds::ExtraFromSeeds(const edm::ParameterSet& iConfig)
    : tracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))) {
  produces<ExtremeLight>();
  produces<TrackingRecHitCollection>();
}

ExtraFromSeeds::~ExtraFromSeeds() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
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
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
