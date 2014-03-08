#ifndef L1TkElectronEtComparator_HH
#define L1TkElectronEtComparator_HH
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

namespace L1TkElectron{
  class EtComparator {
  public:
    bool operator()(const l1extra::L1EmParticle& a, const l1extra::L1EmParticle& b) const {
      double et_a = 0.0;
      double et_b = 0.0;    
      if (cosh(a.eta()) > 0.0) et_a = a.energy()/cosh(a.eta());
      if (cosh(b.eta()) > 0.0) et_b = b.energy()/cosh(b.eta());
      
      return et_a > et_b;
    }
  };
}
#endif
