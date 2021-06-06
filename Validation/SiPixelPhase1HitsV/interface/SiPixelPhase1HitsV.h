#ifndef SiPixelPhase1HitsV_h
#define SiPixelPhase1HitsV_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1HitsV
// Class  :     SiPixelPhase1HitsV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

namespace reco {
  class TrackToTrackingParticleAssociator;
}

class SiPixelPhase1HitsV : public SiPixelPhase1Base {
  enum {
    TOF_R,
    ELOSS,
    ENTRY_EXIT_X,
    ENTRY_EXIT_Y,
    ENTRY_EXIT_Z,
    LOCAL_X,
    LOCAL_Y,
    LOCAL_Z,
    LOCAL_PHI,
    LOCAL_ETA,
    EFFICIENCY_TRACK,
  };

public:
  explicit SiPixelPhase1HitsV(const edm::ParameterSet &conf);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<edm::PSimHitContainer> pixelBarrelLowToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> pixelBarrelHighToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> pixelForwardLowToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> pixelForwardHighToken_;

  edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorByHitsToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
};

#endif
