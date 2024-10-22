#ifndef SiPixelPhase1TrackingParticleV_h
#define SiPixelPhase1TrackingParticleV_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackingParticleV
// Class  :     SiPixelPhase1TrackingParticleV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

namespace reco {
  class TrackToTrackingParticleAssociator;
}

class SiPixelPhase1TrackingParticleV : public SiPixelPhase1Base {
  enum {
    MASS,
    CHARGE,
    ID,
    NHITS,
    MATCHED,
    PT,
    PHI,
    ETA,
    VTX,
    VTY,
    VYZ,
    TIP,
    LIP,
  };

public:
  explicit SiPixelPhase1TrackingParticleV(const edm::ParameterSet &conf);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> vec_TrackingParticle_Token_;
  std::vector<edm::EDGetTokenT<std::vector<PSimHit>>> simHitTokens_;
};

#endif
