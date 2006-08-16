#ifndef SimDataFormats_TrackingParticle_H
#define SimDataFormats_TrackingParticle_H
/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

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
  typedef TrackPSimHitRefToBaseVector::iterator                  pSH_iterator;

//  typedef TrackPSimHitRefVector::iterator 			 pSH_iterator;
//  typedef std::map<int, TrackPSimHitRefVector> 			 TrackIdPSimHitMap;
  
  /// default constructor
  TrackingParticle() { }
  // destructor
  ~TrackingParticle();
  /// constructor from pointer to generator particle
  TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
		    double t, const int pdgId,  const EncodedEventId eventId);
  
  /// PDG id, signal source, crossing number  
  int pdgId() const { return pdgId_; }
  EncodedEventId eventId() const { return eventId_; }
  
  ///iterators
  genp_iterator genParticle_begin() const;
  genp_iterator genParticle_end() const;
  g4t_iterator  g4Track_begin() const;
  g4t_iterator  g4Track_end() const;
  
  pSH_iterator  pSimHit_begin() const;
  pSH_iterator  pSimHit_end() const;

// Setters for G4 and HepMC
  void addG4Track(const SimTrackRef&);
  void addGenParticle(const GenParticleRef&);

  void addPSimHit(const TrackPSimHitRef&);
  
// Getters for Embd and Sim Tracks
  GenParticleRefVector	genParticle() const { return genParticles_; }
  SimTrackRefVector	g4Tracks() const { return g4Tracks_ ; }
 
//  TrackPSimHitRefVector trackPSimHit() const { return trackPSimHit_; }
  TrackPSimHitRefToBaseVector trackPSimHit() const { return trackPSimHit_; }
  
private:
  /// production time
  double t_;
  /// PDG identifier, signal source, crossing number
  int pdgId_;
  EncodedEventId eventId_;
  
  /// references to G4 and HepMC tracks
  SimTrackRefVector g4Tracks_;
  GenParticleRefVector  genParticles_;
 
//  TrackPSimHitRefVector trackPSimHit_;
    TrackPSimHitRefToBaseVector trackPSimHit_;
};

#endif // SimDataFormats_TrackingParticle_H
