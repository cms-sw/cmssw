#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

TrackingParticle::TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
				    double t, int pdgId ) :
  reco::Particle( q, p4, vtx ), t_( t ), pdgId_( pdgId ) {
}

TrackingParticle::~TrackingParticle() { 
}


void TrackingParticle::addGenParticle( const edm::Ref<edm::HepMCProduct, HepMC::GenParticle > &ref) { 
  genParticles_.push_back(ref);
}

void TrackingParticle::addG4Track( const EmbdSimTrackRef &ref) { 
  g4Tracks_.push_back(ref);
}

