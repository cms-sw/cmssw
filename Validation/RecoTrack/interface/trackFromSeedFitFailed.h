#ifndef Validation_RecoTrack_trackFomSeedFitFailed_h
#define Validation_RecoTrack_trackFomSeedFitFailed_h

#include "DataFormats/TrackReco/interface/Track.h"

inline bool trackFromSeedFitFailed(const reco::Track& track) {
  // these magic values denote a case where the fit has failed
  return track.chi2() < 0 && track.ndof() < 0 && track.charge() == 0;
}

#endif
