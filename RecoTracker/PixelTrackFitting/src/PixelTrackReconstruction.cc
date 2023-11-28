#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackCleanerWrapper.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackReconstruction.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

using namespace pixeltrackfitting;
using edm::ParameterSet;

PixelTrackReconstruction::PixelTrackReconstruction(const ParameterSet& cfg, edm::ConsumesCollector&& iC)
    : theHitSetsToken(iC.consumes<RegionsSeedingHitSets>(cfg.getParameter<edm::InputTag>("SeedingHitSets"))),
      theFitterToken(iC.consumes<PixelFitter>(cfg.getParameter<edm::InputTag>("Fitter"))) {
  edm::InputTag filterTag = cfg.getParameter<edm::InputTag>("Filter");
  if (not filterTag.label().empty()) {
    theFilterToken = iC.consumes<PixelTrackFilter>(filterTag);
  }
  std::string cleanerName = cfg.getParameter<std::string>("Cleaner");
  if (not cleanerName.empty()) {
    theCleanerToken = iC.esConsumes(edm::ESInputTag("", cleanerName));
  }
}

PixelTrackReconstruction::~PixelTrackReconstruction() {}

void PixelTrackReconstruction::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("SeedingHitSets", edm::InputTag("pixelTracksHitTriplets"));
  desc.add<edm::InputTag>("Fitter", edm::InputTag("pixelFitterByHelixProjections"));
  desc.add<edm::InputTag>("Filter", edm::InputTag("pixelTrackFilterByKinematics"));
  desc.add<std::string>("Cleaner", "pixelTrackCleanerBySharedHits");
}

void PixelTrackReconstruction::run(TracksWithTTRHs& tracks, edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<RegionsSeedingHitSets> hhitSets;
  ev.getByToken(theHitSetsToken, hhitSets);
  const auto& hitSets = *hhitSets;

  edm::Handle<PixelFitter> hfitter;
  ev.getByToken(theFitterToken, hfitter);
  const auto& fitter = *hfitter;

  const PixelTrackFilter* filter = nullptr;
  if (!theFilterToken.isUninitialized()) {
    edm::Handle<PixelTrackFilter> hfilter;
    ev.getByToken(theFilterToken, hfilter);
    filter = hfilter.product();
  }

  std::vector<const TrackingRecHit*> hits;
  hits.reserve(4);
  for (const auto& regionHitSets : hitSets) {
    const TrackingRegion& region = regionHitSets.region();

    for (const SeedingHitSet& tuplet : regionHitSets) {
      /// FIXME at some point we need to migrate the fitter...
      auto nHits = tuplet.size();
      hits.resize(nHits);
      for (unsigned int iHit = 0; iHit < nHits; ++iHit)
        hits[iHit] = tuplet[iHit];

      // fitting
      std::unique_ptr<reco::Track> track = fitter.run(hits, region);
      if (!track)
        continue;

      if (filter) {
        if (!(*filter)(track.get(), hits)) {
          continue;
        }
      }
      /* roll back to verify if HLT tau is affected  
      // all legacy tracks are "highPurity"
      // setting all others bits as well (as in ckf)
      track->setQuality(reco::TrackBase::loose);
      track->setQuality(reco::TrackBase::tight);
      track->setQuality(reco::TrackBase::highPurity);
      */
      // add tracks
      tracks.emplace_back(track.release(), tuplet);
    }
  }

  // skip ovelrapped tracks
  if (theCleanerToken.isInitialized()) {
    const auto& cleaner = es.getData(theCleanerToken);
    if (cleaner.fast())
      cleaner.cleanTracks(tracks);
    else
      tracks = PixelTrackCleanerWrapper(&cleaner).clean(tracks);
  }
}
