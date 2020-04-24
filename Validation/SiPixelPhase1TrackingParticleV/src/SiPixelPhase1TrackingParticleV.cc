// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackingParticleV
// Class:       SiPixelPhase1TrackingParticleV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "Validation/SiPixelPhase1TrackingParticleV/interface/SiPixelPhase1TrackingParticleV.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class TrackAssociatorByHits; 

namespace {
  bool trackIdHitPairLess(const std::pair<unsigned int, const PSimHit*>& a, const std::pair<unsigned int, const PSimHit*>& b) {
    return a.first < b.first;
  }

  bool trackIdHitPairLessSort(const std::pair<unsigned int, const PSimHit*>& a, const std::pair<unsigned int, const PSimHit*>& b) {
    if(a.first == b.first) {
      const auto atof = edm::isFinite(a.second->timeOfFlight()) ? a.second->timeOfFlight() : std::numeric_limits<decltype(a.second->timeOfFlight())>::max();
      const auto btof = edm::isFinite(b.second->timeOfFlight()) ? b.second->timeOfFlight() : std::numeric_limits<decltype(b.second->timeOfFlight())>::max();
      return atof < btof;
    }
    return a.first < b.first;
  }
}


SiPixelPhase1TrackingParticleV::SiPixelPhase1TrackingParticleV(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig),
  vec_TrackingParticle_Token_( consumes<TrackingParticleCollection>( iConfig.getParameter<edm::InputTag>( "src" ) ) )
{
  for(const auto& tag: iConfig.getParameter<std::vector<edm::InputTag>>("simHitToken")) {
    simHitTokens_.push_back(consumes<std::vector<PSimHit>>(tag));
  }
}

void SiPixelPhase1TrackingParticleV::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<TrackingParticleCollection>  TruthTrackContainer;
  iEvent.getByToken( vec_TrackingParticle_Token_, TruthTrackContainer );
  const TrackingParticleCollection *tPC   = TruthTrackContainer.product();

  // A multimap linking SimTrack::trackId() to a pointer to PSimHit
  // Similar to TrackingTruthAccumulator
  for(const auto& simHitToken: simHitTokens_) {
    edm::Handle<std::vector<PSimHit> > hsimhits;
    iEvent.getByToken(simHitToken, hsimhits);
    trackIdToHitPtr_.reserve(trackIdToHitPtr_.size()+hsimhits->size());
    for(const auto& simHit: *hsimhits) {
      trackIdToHitPtr_.emplace_back(simHit.trackId(), &simHit);
    }
  }
  std::stable_sort(trackIdToHitPtr_.begin(), trackIdToHitPtr_.end(), trackIdHitPairLessSort);


  // Loop over TrackingParticle's
  for (TrackingParticleCollection::const_iterator t = tPC -> begin(); t != tPC -> end(); ++t) {

    // histo manager requires a det ID, use first tracker hit

    bool isBpixtrack = false, isFpixtrack = false;
    DetId id;

    for(const SimTrack& simTrack: t->g4Tracks()) {
      // Logic is from TrackingTruthAccumulator
      auto range = std::equal_range(trackIdToHitPtr_.begin(), trackIdToHitPtr_.end(), std::pair<unsigned int, const PSimHit *>(simTrack.trackId(), nullptr), trackIdHitPairLess);
      if(range.first == range.second) continue;

      auto iHitPtr = range.first;
      for(; iHitPtr != range.second; ++iHitPtr) {
        const PSimHit& simHit = *(iHitPtr->second);
        if(simHit.eventId() != t->eventId())
          continue;
        id = DetId( simHit.detUnitId() );

        // check we are in pixel
        uint32_t subdetid = (id.subdetId());
	if (subdetid == PixelSubdetector::PixelBarrel) isBpixtrack = true;
	if (subdetid == PixelSubdetector::PixelEndcap) isFpixtrack = true;
	if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap) continue;
      }
    }

    if ( isBpixtrack || isFpixtrack ) {
      histo[MASS].fill(t->mass(), id, &iEvent);
      histo[CHARGE].fill(t->charge(), id, &iEvent);
      histo[ID].fill(t->pdgId(), id, &iEvent);
      histo[NHITS].fill(t->numberOfTrackerHits(), id, &iEvent);
      histo[MATCHED].fill(t->numberOfTrackerLayers(), id, &iEvent);
      histo[PT].fill(sqrt(t->momentum().perp2()), id, &iEvent);
      histo[PHI].fill(t->momentum().Phi(), id, &iEvent);
      histo[ETA].fill(t->momentum().eta(), id, &iEvent);
      histo[VTX].fill(t->vx(), id, &iEvent);
      histo[VTY].fill(t->vy(), id, &iEvent);
      histo[VYZ].fill(t->vz(), id, &iEvent);
      histo[TIP].fill(sqrt(t->vertex().perp2()), id, &iEvent);
      histo[LIP].fill(t->vz(), id, &iEvent);
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackingParticleV);

