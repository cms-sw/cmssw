#ifndef Validation_EventGenerator_GenPtcValidationHelper
#define Validation_EventGenerator_GenPtcValidationHelper

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <vector>

namespace GenPtcValidationHelper {
  template <class T>

  static bool sortByPt(const T lhs, const T rhs) { return lhs->pt() > rhs->pt(); }
  bool isFinalStateLepton(const reco::GenParticleRef ptc);
  void findFSRPhotons(const std::vector<reco::GenParticleRef>& leps, const reco::GenParticleCollection& ptcls,
    double dRcut, std::vector<reco::GenParticle>& FSRphotons);

}

#endif
