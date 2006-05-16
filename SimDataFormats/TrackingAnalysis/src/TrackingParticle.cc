#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

TrackingParticle::TrackingParticle( Charge q, const LorentzVector & p4, const Point & vtx,
				    int pdgId ) :
  reco::Particle( q, p4, vtx ), pdgId_( pdgId ) {
}

TrackingParticle::~TrackingParticle() { 
}
