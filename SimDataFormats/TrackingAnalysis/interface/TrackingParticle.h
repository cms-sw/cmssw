#ifndef SimDataFormats_TrackingParticle_H
#define SimDataFormats_TrackingParticle_H
/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

// Should move these into EmbdSimTrackContainer.h, no time before 8.0-pre2

//#include "DataFormats/Common/interface/Ref.h"
//#include "DataFormats/Common/interface/RefProd.h"
//#include "DataFormats/Common/interface/RefVector.h"
//typedef edm::Ref<edm::EmbdSimTrackContainer> EmbdSimTrackRef;
//typedef edm::RefProd<edm::EmbdSimTrackContainer> EmbdSimTrackRefProd;
//typedef edm::RefVector<edm::EmbdSimTrackContainer> EmbdSimTrackRefVector;
using edm::EmbdSimTrackRef;
using edm::EmbdSimTrackRefProd;
using edm::EmbdSimTrackRefVector;

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
		    double t, int pdgId );
  /// PDG identifier  
  int pdgId() const { return pdgId_; }
  /// reference to G4 track
//  const  EmbdSimTrackRef & g4Track() const { return g4Track_; }
  /// pointer to generator particle
//  int genParticle() const { return genParticle_; }
  /// set reference to G4 track
//  void setG4Track( const EmbdSimTrackRef & r ) { g4Track_ = r; }
  void addG4Track(const EmbdSimTrackRef&);
  /// set pointer to generator particle
//  void setGenParticle( int gp ) { genParticle_ = gp; }
  void addGenParticle(const GenParticleRef&);

private:
  /// production time
  double t_;
  /// PDG identifier
  int pdgId_;
  /// references to G4 and HepMC tracks
  EmbdSimTrackRefVector g4Tracks_;
  GenParticleRefVector  genParticles_;
};

#endif // SimDataFormats_TrackingParticle_H
