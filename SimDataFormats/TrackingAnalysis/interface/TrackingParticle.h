#ifndef SimDataFormats_TrackingParticle_H
#define SimDataFormats_TrackingParticle_H
/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"

namespace HepMC {
  class GenParticle;
}

class TrackingParticle : public reco::Particle {
public:
  /// reference to HepMC::GenParticle
  typedef const HepMC::GenParticle * GenParticleRef;
  /// default constructor
  TrackingParticle() { }
  // destructor
  ~TrackingParticle();
  /// constructor from pointer to generator particle
  TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
		    int pdgId );
  /// PDG identifier  
  int pdgId() const { return pdgId_; }
  /// reference to G4 track
  const  EmbdSimTrackRef & g4Track() const { return g4Track_; }
  /// pointer to generator particle
  GenParticleRef genParticle() const { return genParticle_; }
  /// set reference to G4 track
  void setG4Track( const EmbdSimTrackRef & r ) { g4Track_ = r; }
  /// set pointer to generator particle
  void setGenParticle( GenParticleRef r ) { genParticle_ = r; }

private:
  /// PDG identifier
  int pdgId_;
  /// reference to G4 track
  EmbdSimTrackRef g4Track_;
  /// pointer to generator particle
  GenParticleRef genParticle_;
};

#endif // SimDataFormats_TrackingParticle_H
