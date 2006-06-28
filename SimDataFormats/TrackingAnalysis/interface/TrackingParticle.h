#ifndef SimDataFormats_TrackingParticle_H
#define SimDataFormats_TrackingParticle_H
/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

namespace HepMC {
  class GenParticle;
}

class TrackingParticle : public reco::Particle {
public:
  /// reference to HepMC::GenParticle
  typedef edm::RefVector<edm::HepMCProduct, HepMC::GenParticle > GenParticleRefVector;
  typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle >       GenParticleRef;
  /// default constructor
  TrackingParticle() { }
  // destructor
  ~TrackingParticle();
  /// constructor from pointer to generator particle
  TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
		    double t, const int pdgId,  const int source, const int crossing );
  /// PDG identifier  
  int pdgId() const { return pdgId_; }
  int source() const { return signalSource_ % 4 == 0; }
  int crossing() const { return crossing_; }

// Setters for Embd and Sim Tracks
  void addG4Track(const SimTrackRef&);
  void addGenParticle(const GenParticleRef&);
  
// Need Getters for Embd and Sim Tracks

private:
  /// production time
  double t_;
  /// PDG identifier
  int pdgId_;
  /// references to G4 and HepMC tracks
  SimTrackRefVector g4Tracks_;
  GenParticleRefVector  genParticles_;
  int signalSource_; 
  int crossing_;
};

#endif // SimDataFormats_TrackingParticle_H
