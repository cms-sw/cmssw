#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

TrackingParticle::TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
				    double t, const int pdgId, const int source, const int crossing ) :
  reco::Particle( q, p4, vtx ), t_( t ), pdgId_( pdgId ), signalSource_( source ), crossing_( crossing ){
}

TrackingParticle::~TrackingParticle() { 
}


void TrackingParticle::addGenParticle( const edm::Ref<edm::HepMCProduct, HepMC::GenParticle > &ref) { 
  genParticles_.push_back(ref);
}

void TrackingParticle::addG4Track( const SimTrackRef &ref) { 
  g4Tracks_.push_back(ref);
}

//int TrackingParticle::source(){}
