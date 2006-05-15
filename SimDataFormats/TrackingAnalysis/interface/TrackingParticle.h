#ifndef SimDataFormats_TrackingParticle_H
#define SimDataFormats_TrackingParticle_H
/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

namespace HepMC {
  class GenParticle;
}

class TrackingParticle  {
public:
  /// reference to HepMC::GenParticle
  typedef const HepMC::GenParticle * GenParticleRef;
  /// default constructor
  TrackingParticle() { }
  /// constructor from pointer to generator particle
  TrackingParticle( const HepMC::GenParticle * );
  /// pointer to generator particle
  GenParticleRef genParticle() const { return genParticle_; }
  /// PDG identifier  
  int pdgId() const { return pdgId_; }

private:
  /// pointer to generator particle
  GenParticleRef genParticle_;
  /// PDG identifier
  int pdgId_;
};

#endif // SimDataFormats_TrackingParticle_H
