#ifndef L1TkTauEtComparator_HH
#define L1TkTauEtComparator_HH
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"

namespace L1TkTau{
  class EtComparator {
  public:
    bool operator()(const l1extra::L1TkTauParticle& a, const l1extra::L1TkTauParticle& b) const {
      double et_a = a.et();
      double et_b = b.et();
      return et_a > et_b;
    }
  };

  class PtComparator {
  public:
    bool operator()(const l1extra::L1TkTauParticle& a, const l1extra::L1TkTauParticle& b) const {
      double et_a = a.pt();
      double et_b = b.pt();
      return et_a > et_b;
    }
  };

}
#endif

