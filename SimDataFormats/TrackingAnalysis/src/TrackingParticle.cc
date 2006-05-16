#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

TrackingParticle::TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
				    double t, int pdgId ) :
  reco::Particle( q, p4, vtx ), t_( t ), pdgId_( pdgId ) {
}

TrackingParticle::~TrackingParticle() { 
}
