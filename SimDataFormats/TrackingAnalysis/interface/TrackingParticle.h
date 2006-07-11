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
  typedef GenParticleRefVector::iterator		         genp_iterator;
  typedef SimTrackRefVector::iterator				 g4t_iterator;
 
  /// default constructor
  TrackingParticle() { }
  // destructor
  ~TrackingParticle();
  /// constructor from pointer to generator particle
  TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
		    double t, const int pdgId,  const int source, const int crossing );
  
  /// PDG id, signal source, crossing number  
  int pdgId() const { return pdgId_; }
  int source() const { return signalSource_ % 4 == 0; }
  int crossing() const { return crossing_; }
  
  ///iterators
  genp_iterator genParticle_begin() const;
  genp_iterator genParticle_end() const;
  g4t_iterator  g4Track_begin() const;
  g4t_iterator  g4Track_end() const;
  

// Setters for G4 and HepMC
  void addG4Track(const SimTrackRef&);
  void addGenParticle(const GenParticleRef&);
  
// Getters for Embd and Sim Tracks
  GenParticleRefVector	genParticle() const { return genParticles_; }
  SimTrackRefVector	g4Tracks() const { return g4Tracks_ ; }
  
private:
  /// production time
  double t_;
  /// PDG identifier, signal source, crossing number
  int pdgId_;
  int signalSource_; 
  int crossing_;
  /// references to G4 and HepMC tracks
  SimTrackRefVector g4Tracks_;
  GenParticleRefVector  genParticles_;
};

#endif // SimDataFormats_TrackingParticle_H
